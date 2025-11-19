import pandas as pd
import re

dollar_step = 8.1
fee2x = 2
# fee2x = 0.36

futures_fee_funcs = {
    'base': lambda total,count: total - count*fee2x,
    r'.*BR..*': lambda total,count: total*100*dollar_step - count*fee2x,
    r'.*ED.*': lambda total,count: total*10000*dollar_step - count*fee2x,
    r'.*EURRUBF.*': lambda total,count: total*1000 - count*fee2x,
    r'.*IMOEXF': lambda total,count: total*10 - count*fee2x,
    r'.*MM..*': lambda total,count: total*10 - count*fee2x,
    r'.*NG..*': lambda total,count: total*1000*dollar_step - count*fee2x,
    r'.*RM..*': lambda total,count: total*2*dollar_step - count*fee2x,
    r'.*RI..*': lambda total,count: total*2*dollar_step*0.1 - count*fee2x,
    r'.*CNYRUBF.*': lambda total,count: total*1000 - count*fee2x,
    r'.*CR..*': lambda total,count: total*1000 - count*fee2x,
    r'.*GD..*': lambda total,count: total*10*dollar_step - count*fee2x,
    r'.*USDRUBF.*': lambda total,count: total*1000 - count*fee2x,
    r'.*SV..*': lambda total,count: total*100*dollar_step - count*fee2x,
    r'.*PD..*': lambda total,count: total*10*dollar_step - count*fee2x,
    r'.*PT..*': lambda total,count: total*10*dollar_step - count*fee2x,
    r'.*UC..*': lambda total,count: total*1000*10.94 - count*fee2x,
    r'.*SF..*': lambda total,count: total*10*dollar_step - count*fee2x,
    r'.*NA..*': lambda total,count: total*dollar_step*0.1 - count*fee2x,
    r'.*CC..*': lambda total,count: total*10 - count*fee2x,
    r'.*SBERF.*': lambda total,count: total*100 - count*fee2x,
    r'.*GAZPF.*': lambda total,count: total*100 - count*fee2x,
    r'.*IB..*': lambda total,count: total*10*dollar_step - count*fee2x,
}

# tests = ('BR','BRQ5','5BRQ5','5_BRQ5','BQR5','BRU5','BRQ51')
def get_func_vtb_fee(name):
    for fff in futures_fee_funcs:
        if re.match(fff,name):
            return futures_fee_funcs[fff]
    return futures_fee_funcs['base']

# for t in tests:
#     print(t,get_func_vtb_fee(t)(10,10))
# print(get_func_vtb_fee('RMU5'))