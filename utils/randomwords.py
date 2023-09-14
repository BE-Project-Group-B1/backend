import random
list = ["steel","pink","around","mouse","sky","orange","apple","branch","green","store","garden","silver","purple","month","wolf","open","dangerous","marathon"]
 

def genWords(n): 
    return ' '.join(random.sample(list,n))
