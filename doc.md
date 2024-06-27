### 交易分量文档

#### 1. 背景

在银行间市场，公司会从管理的基金中拿出一部分资金用于同业拆借交易，比如基金A拿出10亿、基金B拿出20亿、基金C拿出30亿等，与此同时，不同的借款方会有金额不等的借款需求，比如机构甲需要15亿(2.1%)、机构乙需要38亿(2.05%)等，不同机构承诺的利率不同。参与银行间市场的资金规模大，涉及到的基金产品和交易对手较广，交易部门需要审慎设计交易方案，匹配出借方与借款方的需求，保障交易顺利进行。设计交易方案也称作交易分量。

模型输入:  
出借方资金（货币基金）：$\textbf{f} = (f_1, f_2, ..., f_m)^T,$  
借入方所需资金（交易对手）：$\textbf{i} = (i_1, i_2,..., i_n)^T,$  
借入方承诺利率：$\textbf{R} = (R_1, R_2,..., R_n)^T,$  
借入方的人工分户指示：$\textbf{MANU} = (manu_1, manu_2,..., manu_n)^T,$  
借入方的错户指示：$\textbf{FORB} = (forb_1, forb_2,..., forb_n)^T,$

人工分户指示和错户指示默认值是-1（无指示），若为0到m-1的整数则表示人工分户指示或错户指示，0到m-1的整数对应出借方序号。
测试用例可供参考。

模型输出：  
如上输入所示，共有$m$个出借方基金，有$n$个借入方交易对手。交易方案可由矩阵表示
$
\begin{align}
\mathbf{H} &= \begin{pmatrix}
h_{ij}\nonumber
\end{pmatrix}_{m\times n}
\end{align}
$
其中$h_{ij}$就表示第$i$个基金产品和第$j$个对手方的交易额。

#### 2. 交易分量的约束条件
具体的约束条件及非功能性要求，可以参考交易分量代码中的README.md文档.  
在众多约束中，几个核心约束是加权利率公允、单笔订单勿拆太散、考虑错户需求、分拆后订单金额不得过小、求解时间短等.

#### 3. 交易分量的技术方案
具体的技术思路可以参考专利交底书。本版交易分量模型的主要功能包装成allocation_agent_v2类，使用时实例化。下面的函数是主要的执行函数，函数的输入即前文的模型输入，函数的输出除前文的模型输出外，还包括实例化的agent. 输出agent的原因是因为agent的部分属性与函数将用来做后续的效果评估。

下面就主体函数来介绍每个函数在整体模型的用途，具体函数的细节可见service/agent.py, 每个函数都包含详细的注释。

```python
def op_excel(df: pd.DataFrame):
    """
    主函数！！！
    :param file: 上传的数据文件, 其中包含了fund, insti, rate, manual, forbidden列
    :return result: 返回处理后的结果
    :return distribution_processor: 返回处理器, 以便后续调用分析性能等
    """
    distribution_processor = allocation_agent_v2(
        df, lambda_value=0.3, sparse_coeffi=1e5, fair_coeffi=0
    )
    # 首先做预处理来做降维，以提升后续的处理效率。即先按一些启发式方法分配交易，分配的结果会通过agent的属性来传递
    # forbidden_preprocess()用来记录错户指示
    distribution_processor.forbidden_preprocess()
    # manual_preprocess()用来分配人工分户指示
    distribution_processor.manual_preprocess()
    # allo_preprocess()用于最初步的降维
    distribution_processor.allo_preprocess()
    # forbidden_allo()用于启发式地分配带错户指示的对手方交易，降维
    distribution_processor.forbidden_allo()
    # 仍然是降维
    allo, fundp = distribution_processor.allo_insti_into_fund()
    # 经降维后，出借方基金的维数预计会大幅下降，此时如果只有一家基金没分完，则直接分配；如多家基金没分完，则进入专利交底书所写的2步优化
    # check the number of non-zero elements in dfp['fund']
    fund_left_num = np.count_nonzero(fundp)

    # 1st step
    if fund_left_num > 1:
        # 进入优化程序
        result = distribution_processor.optimize_sparse()
    else:
        # 直接分配
        result = distribution_processor.onefund_plan()
    
    # 2nd step
    # 评估公允性
    rate_f, rate_max_min = distribution_processor.result_assessment(result)
    # 微调
    result = distribution_processor.fine_tune_step_by_step()
    # 后续微调
    result1 = distribution_processor.fix_lower_than_a_half_after_nonlinear_opti()

    if result1 is not None:
        result = result1
    return result, distribution_processor
```

#### 4. 交易分量结果的评估
由于交易分量本身的特点，约束条件较多，需要较多的评估指标。接下来依次介绍各指标的计算逻辑。计算各指标的具体代码可见service/ui.py, 其中较复杂的计算包含了详细的注释。
请注意，此处的评估仅供测试参考！最终的评估仍需经业务部门判断

```python
        [   output_dataframe, # 模型输出
            max_min_diff,     # 加权利率的极差，先计算各基金加权利率，取其中最大和最小做差
            num_of_4ormore,   # 根据模型输出，判断其中有多少个借款方被拆成4笔及以上
            num_of_3ormore,   # 根据模型输出，判断其中有多少个借款方被拆成3笔及以上
            num_of_small,     # 根据模型输出，判断其中是否有拆分后小于0.5e的订单
            time_used,        # 花费时间
            num_nonzero,      # 模型输出里非零元素个数
            num_total,        # 模型输出总共的元素个数
            cuohu,            # 评估模型输出是否满足错户指示，只需遍历forbidden列的错户指示，check模型输出对应的位置是否为0即可
            rengong,          # 评估模型输出是否满足人工分户指示，按照manual列check模型输出的对应位置
            error_text        # 报错信息
        ]
```

#### 5. 总结与展望
目前，本版模型已经可以覆盖大部分的业务场景，通过业务测评。

README.md文档提示了本版模型仍然存在的一些问题，可供参考；  
比较突出的问题是当前版本仍不支持“对手方包含多个错户产品”的情形，可以通过修改本版模型的部分函数来处理，更佳的办法是瞄准“对手方包含多个错户产品”的情形，针对性地做研发。