import json
import math
import torch
import torch.distributions as distributions
from daphne import daphne
# funcprimitives
from tests import is_tol, run_prob_test,load_truth
# Useful functions
from primitives import _hashmap, _vector, _totensor
from primitives import _put, _remove, _append, _get
from primitives import _squareroot, _mat_repmat, _mat_transpose
# Dist
from distributions import Normal, Bernoulli, Categorical, Dirichlet, Gamma

# OPS
basic_ops = {'+':torch.add,
             '-':torch.sub,
             '*':torch.mul,
             '/':torch.div
}

math_ops = {'sqrt': lambda x: _squareroot(x)
}

data_struct_ops = {'vector': lambda x: _vector(x),
                   'hash-map': lambda x: _hashmap(x)
}

data_interact_ops = {'first': lambda x: x[0],      # retrieves the first element of a list or vector e
                     'second': lambda x: x[1],     # retrieves the second element of a list or vector e
                     'last': lambda x: x[-1],      # retrieves the last element of a list or vector e
                     'rest': lambda x: x[1:],      # retrieves the rest of the element of a list except the first one
                     'get': lambda x, idx: _get(x, idx),              # retrieves an element at index e2 from a list or vector e1, or the element at key e2 from a hash map e1.
                     'append': lambda x, y: _append(x, y),            # (append e1 e2) appends e2 to the end of a list or vector e1
                     'remove': lambda x, idx: _remove(x, idx),        # (remove e1 e2) removes the element at index/key e2 with the value e2 in a vector or hash-map e1.
                     'put': lambda x, idx, value: _put(x, idx, value) # (put e1 e2 e3) replaces the element at index/key e2 with the value e3 in a vector or hash-map e1.
}

cond_ops={"<": lambda a, b: a < b,
          ">": lambda a, b: a > b,
          "=": lambda a, b: a == b,
          ">=": lambda a, b: a >= b,
          "<=": lambda a, b: a <= b,
          "or": lambda a, b: a or b,
          "and": lambda a, b: a and b
}

nn_ops={"mat-tanh": lambda a: torch.tanh(a),
        "mat-add": lambda a, b: torch.add(a, b),
        "mat-mul": lambda a, b: torch.matmul(a, b),
        "mat-repmat": lambda a, b, c: _mat_repmat(a, b, c),
        "mat-transpose": lambda a: _mat_transpose(a)
}


# Global vars
global rho;
rho = {}
DEBUG = False # Set to true to see intermediate outputs for debugging purposes
#----------------------------Evaluation Functions -----------------------------#
def eval(e, sig, l):
    # Empty list
    if not e:
        return [False, sig]

    if DEBUG:
        print('Current E: ', e)

    # import pdb; pdb.set_trace()
    if len(e) == 1:
        # Check if a single string ast [['mu']]
        single_val = False
        if isinstance(e[0], str):
            root = e[0]
            tail = []
            single_val = True
        else:
            e = e[0]

        if DEBUG:
            print('Current program: ', e)
        try:
            # Check if a single string such as ast = ['mu']
            if not single_val:
                if len(e) == 1:
                    if isinstance(e[0], str):
                        root = e[0]
                        tail = []
                else:
                    root, *tail = e
            if DEBUG:
                print('Current OP: ', root)
                print('Current TAIL: ', tail)

            # Sample
            if root == 'sample':
                if DEBUG:
                    print('Sampler program: ', tail)
                sampler, sig = evaluate_program(tail, sig=sig, l=l)
                if DEBUG:
                    print('Sampler: ', sampler)

                if root not in Q.keys():
                    sigma[Q[root]] = sampler

                try:
                    c = sigma[Q[root]].sample()
                except:
                    # For some reason if it is not a sampler object
                    raise AssertionError('Failed to sample!')

                try:
                    sigma = sigma[Q[root]].sample()
                except:
                    # For some reason if it is not a sampler object
                    raise AssertionError('Failed to sample!')

                sigma[G[root]] = sigma[Q[root]].sample()
                logWv = sampler.log_prob(c) - sigma[Q[root]].log_prob(c)
                try:
                    if "logW" in sig.keys():
                        sig["logW"] += sampler.log_prob(c)
                    else:
                        sig["logW"]  = sampler.log_prob(c)
                except:
                    if "logW" in sig.keys():
                        sig["logW"] += 0.0
                    else:
                        sig["logW"]  = 0.0
                return [c, sig]

            # Observe
            elif root == 'observe':
                # import pdb; pdb.set_trace()
                if DEBUG:
                    print('Observe tail: ', tail)
                    print('Observe tail length: ', len(tail))

                if len(tail) == 2:
                    # Check for single referenced string
                    if isinstance(tail[0], str):
                        ob_pm1 = [tail[0]]
                    else:
                        ob_pm1 = tail[0]
                    if isinstance(tail[1], str):
                        ob_pm2 = [tail[1]]
                    else:
                        ob_pm2 = tail[1]
                else:
                    raise AssertionError('Unknown list of observe params!')
                if DEBUG:
                    print('Observe Param-1: ', ob_pm1)
                    print('Observe Param-2: ', ob_pm2)

                # Evaluate observe params
                p, sig = evaluate_program([ob_pm1], sig=sig, l=l)
                c, sig = evaluate_program([ob_pm2], sig=sig, l=l)
                value = _totensor(x=value)
                if DEBUG:
                    print('Observe distribution: ', distn)
                    print('Observe Value: ', value, "\n")

                try:
                    if "logW" in sig.keys():
                        sig["logW"] += distn.log_prob(value)
                    else:
                        sig["logW"]  = distn.log_prob(value)
                except:
                    if "logW" in sig.keys():
                        sig["logW"] += 0.0
                    else:
                        sig["logW"]  = 0.0
                return [c, sig]

            # Get distribution
            elif root in dist_ops.keys():
                 # import pdb; pdb.set_trace()
                op_func = dist_ops[root]
                if len(tail) == 2:
                    # Check for single referenced string
                    if isinstance(tail[0], str):
                        param1 = [tail[0]]
                    else:
                        param1 = tail[0]
                    if isinstance(tail[1], str):
                        param2 = [tail[1]]
                    else:
                        param2 = tail[1]
                    if DEBUG:
                        print('Sampler Parameter-1: ', param1)
                        print('Sampler Parameter-2: ', param2)
                    # Eval params
                    para1, sig = evaluate_program([param1], sig=sig, l=l)
                    # Make sure to have it in torch tensor
                    try:
                        para1 = _totensor(x=para1)
                    except:
                        # Most likely a tensor inside a list
                        if isinstance(para1, list):
                            para1 = para1[0]
                            para1 = _totensor(x=para1)
                    para2, sig = evaluate_program([param2], sig=sig, l=l)
                    try:
                        para2 = _totensor(x=para2)
                    except:
                        # Most likely a tensor inside a list
                        if isinstance(para2, list):
                            para2 = para2[0]
                            para2 = _totensor(x=para2)
                    if DEBUG:
                        print('Eval Sampler Parameter-1: ', para1)
                        print('Eval Sampler Parameter-2: ', para2, "\n")
                    return [op_func(para1, para2), sig]
                else:
                    # Exponential has only one parameter
                    # Check for single referenced string
                    if isinstance(tail[0], str):
                        param1 = [tail[0]]
                    else:
                        param1 = tail[0]
                    if DEBUG:
                        print('Sampler Parameter-1: ', param1)
                    para1, sig = evaluate_program([param1], sig=sig, l=l)
                    if DEBUG:
                        print('Eval Sampler Parameter-1: ', para1)
                    # Make sure to have it in torch tensor
                    try:
                        para1 = _totensor(x=para1)
                    except:
                        # Most likely a tensor inside a list
                        if isinstance(para1, list):
                            para1 = para1[0]
                            para1 = _totensor(x=para1)
                    if DEBUG:
                        print('Tensor Sampler Parameter-1: ', para1, "\n")
                    return [op_func(para1), sig]

            # Basic primitives
            elif root in basic_ops.keys():
                op_func = basic_ops[root]
                eval_1, sig = evaluate_program([tail[0]], sig=sig, l=l)
                 # Make sure in floating point
                if torch.is_tensor(eval_1):
                    eval_1 = eval_1.type(torch.float32)
                elif isinstance(eval_1, int):
                    eval_1 = float(eval_1)
                # Make sure not list, if evals returned as list
                elif isinstance(eval_1, list):
                    eval_1 = eval_1[0]
                # Evalute tail
                eval_2, sig = evaluate_program(tail[1:], sig=sig, l=l)
                 # Make sure in floating point
                if torch.is_tensor(eval_2):
                    eval_2 = eval_2.type(torch.float32)
                elif isinstance(eval_2, int):
                    eval_2 = float(eval_2)
                # Make sure not list, if evals returned as list
                if isinstance(eval_2, list):
                    eval_2 = eval_2[0]
                if DEBUG:
                    print('Basic OP eval-1: ', eval_1)
                    print('Basic OP eval-2: ', eval_2)
                op_eval = op_func(eval_1, eval_2)
                return [op_eval, sig]

            # Math ops
            elif root in math_ops.keys():
                op_func = math_ops[root]
                op_eval = op_func(tail)
                return [op_eval, sig]

            # NN ops
            elif root in nn_ops.keys():
                # import pdb; pdb.set_trace()
                op_func = nn_ops[root]
                if root == "mat-add" or root == "mat-mul":
                    e1, e2 = tail
                    # Operand-1
                    if isinstance(e1, list) and len(e1) == 1:
                        a, sig = evaluate_program(e1, sig=sig, l=l)
                    elif isinstance(e1, list):
                        a, sig = evaluate_program([e1], sig=sig, l=l)
                    else:
                        # Most likely a pre-defined varibale in l
                        a = l[e1]
                    # Operand-2
                    if isinstance(e2, list) and len(e2) == 1:
                        b, sig = evaluate_program(e2, sig=sig, l=l)
                    elif isinstance(e2, list):
                        b, sig = evaluate_program([e2], sig=sig, l=l)
                    else:
                        b = l[e2] # Most likely a pre-defined varibale in l
                    if DEBUG:
                        print('Evaluated MatMul-1: ', a)
                        print('Evaluated MatMul-2: ', b)
                    # OP
                    return [op_func(a, b), sig]
                # ["mat-repmat", "b_0", 1, 5]
                elif root == "mat-repmat":
                    e1, e2, e3 = tail
                    # Initial MAT
                    if isinstance(e1, list) and len(e1) == 1:
                        a, sig = evaluate_program(e1, sig=sig, l=l)
                    elif isinstance(e1, list):
                        a, sig = evaluate_program([e1], sig=sig, l=l)
                    else:
                        a = l[e1] # Most likely a pre-defined varibale in l
                    # Repeat axis 1
                    if isinstance(e2, list) and len(e2) == 1:
                        b, sig = evaluate_program(e2, sig=sig, l=l)
                    elif isinstance(e2, list):
                        b, sig = evaluate_program([e2], sig=sig, l=l)
                    elif isinstance(e2, float) or isinstance(e2, int):
                        b = int(e2)
                    else:
                        b = l[e2] # Most likely a pre-defined varibale in l
                    # Repeat axis 2
                    if isinstance(e3, list) and len(e3) == 1:
                        c, sig = evaluate_program(e3, sig=sig, l=l)
                    elif isinstance(e3, list):
                        c, sig = evaluate_program([e3], sig=sig, l=l)
                    elif isinstance(e3, float) or isinstance(e3, int):
                        c = int(e3)
                    else:
                        c = l[e3] # Most likely a pre-defined varibale in l
                    # OP
                    return [op_func(a, b, c), sig]
                else:
                    e1 = tail
                    if isinstance(e1, list) and len(e1) == 1:
                        a, sig = evaluate_program(e1, sig=sig, l=l)
                    elif isinstance(e1, list):
                        a, sig = evaluate_program([e1], sig=sig, l=l)
                    else:
                        a = l[e1] # Most likely a pre-defined varibale in l
                    if DEBUG:
                        print('Evaluated Matrix: ', a)
                    # OP
                    return [op_func(a), sig]

            # Data structures-- Vector
            elif root == "vector":
                # import pdb; pdb.set_trace()
                op_func = data_struct_ops[root]
                if DEBUG:
                    print('Data Structure data: ', tail)
                # Eval tails:
                tail_data = torch.zeros(0, dtype=torch.float32)
                for T in range(len(tail)):
                    # Check for single referenced string
                    if isinstance(tail[T], str):
                        VT = [tail[T]]
                    else:
                        VT = tail[T]
                    if DEBUG:
                        print('Pre-Evaluated Data Structure data: ', VT)
                    eval_T, sig = evaluate_program([VT], sig, l=l)
                    if DEBUG:
                        print('Evaluated Data Structure data: ', eval_T)
                    # If sample object then take a sample
                    try:
                        eval_T = eval_T.sample()
                    except:
                        pass
                    # Check if not torch tensor
                    if not torch.is_tensor(eval_T):
                        if isinstance(eval_T, list):
                            eval_T = torch.tensor(eval_T, dtype=torch.float32)
                        else:
                            eval_T = torch.tensor([eval_T], dtype=torch.float32)
                    # Check for 0 dimensional tensor
                    elif eval_T.shape == torch.Size([]):
                        eval_T = torch.tensor([eval_T.item()], dtype=torch.float32)
                    # Concat
                    try:
                        tail_data = torch.cat((tail_data, eval_T))
                    except:
                        raise AssertionError('Cannot append the torch tensors')
                if DEBUG:
                    print('Eval Data Structure data: ', tail_data)
                return [tail_data, sig]

            # Data structures-- hash-map
            elif root == "hash-map":
                op_func = data_struct_ops[root]
                return [op_func(tail), sig]

            # Data structures interaction
            elif root in data_interact_ops.keys():
                op_func = data_interact_ops[root]
                # ['put', ['vector', 2, 3, 4, 5], 2, 3]
                if root == 'put':
                    e1, e2, e3 = tail
                    if isinstance(e1, list):
                        get_data_struct, sig = evaluate_program([e1], sig=sig, l=l)
                    else:
                        # Most likely a pre-defined varibale in l
                        get_data_struct = l[e1]
                    # Get index
                    if isinstance(e2, list):
                        e2_idx, sig = evaluate_program([e2], sig=sig, l=l)
                    elif isinstance(e2, float) or isinstance(e2, int):
                        e2_idx = int(e2)
                    else:
                        # Most likely a pre-defined varibale in l
                        e2_idx = l[e2]
                    # Get Value
                    if isinstance(e3, list):
                        e3_val, sig = evaluate_program([e3], sig=sig, l=l)
                    elif isinstance(e3, float) or isinstance(e3, int):
                        e3_val = e3
                    else:
                        # Most likely a pre-defined varibale in l
                        e3_val = l[e3]
                    if DEBUG:
                        print('Data : ', get_data_struct)
                        print('Index: ', e2_idx)
                        print('Value: ', e3_val)
                    return [op_func(get_data_struct, e2_idx, e3_val), sig]
                # ['remove'/'get', ['vector', 2, 3, 4, 5], 2]
                elif root == 'remove' or root == 'get':
                    # import pdb; pdb.set_trace()
                    e1, e2 = tail
                    if DEBUG:
                        print('e1: ', e1)
                        print('e2: ', e2)
                    if isinstance(e1, list):
                        get_data_struct, sig = evaluate_program([e1], sig=sig, l=l)
                    else:
                        # Most likely a pre-defined varibale in l
                        get_data_struct = l[e1]
                    if isinstance(e2, list):
                        e2_idx, sig = evaluate_program([e2], sig=sig, l=l)
                    elif isinstance(e2, float) or isinstance(e2, int):
                        e2_idx = e2
                    else:
                        # Otherwise Most likely a pre-defined varibale in l
                        e2_idx = l[e2]
                        if isinstance(e2_idx, list):
                            e2_idx = e2_idx[0]
                    if DEBUG:
                        print('Data : ', get_data_struct)
                        print('Index/Value: ', e2_idx)
                    # Convert index to type-int
                    if torch.is_tensor(e2_idx):
                        e2_idx = e2_idx.long()
                    else:
                        e2_idx = int(e2_idx)
                    return [op_func(get_data_struct, e2_idx), sig]
                # ['append', ['vector', 2, 3, 4, 5], 2]
                elif root == 'append':
                    # import pdb; pdb.set_trace()
                    get_list1, get_list2 = tail
                    # Evalute exp1
                    if isinstance(get_list1, list):
                        get_data_eval_1, sig = evaluate_program([get_list1], sig=sig, l=l)
                    elif isinstance(get_list1, float) or isinstance(get_list1, int):
                        get_data_eval_1 = get_list1
                    else:
                        get_data_eval_1 = l[get_list1] # Most likely a pre-defined varibale in l
                    if DEBUG:
                        print('Op Eval-1: ', get_data_eval_1)
                    # Evalute exp2
                    if isinstance(get_list2, list):
                        get_data_eval_2, sig = evaluate_program([get_list2], sig=sig, l=l)
                    elif isinstance(get_list2, float) or isinstance(get_list2, int):
                        get_data_eval_2 = get_list2
                    else:
                        get_data_eval_2 = l[get_list2] # Most likely a pre-defined varibale in l
                    if DEBUG:
                        print('Op Eval-2: ', get_data_eval_2)
                    # Check if not torch tensor
                    if not torch.is_tensor(get_data_eval_1):
                        if isinstance(get_data_eval_1, list):
                            get_data_eval_1 = torch.tensor(get_data_eval_1, dtype=torch.float32)
                        else:
                            get_data_eval_1 = torch.tensor([get_data_eval_1], dtype=torch.float32)
                    # Check for 0 dimensional tensor
                    elif get_data_eval_1.shape == torch.Size([]):
                        get_data_eval_1 = torch.tensor([get_data_eval_1.item()], dtype=torch.float32)
                    # Check if not torch tensor
                    if not torch.is_tensor(get_data_eval_2):
                        if isinstance(get_data_eval_2, list):
                            get_data_eval_2 = torch.tensor(get_data_eval_2, dtype=torch.float32)
                        else:
                            get_data_eval_2 = torch.tensor([get_data_eval_2], dtype=torch.float32)
                    # Check for 0 dimensional tensor
                    elif get_data_eval_2.shape == torch.Size([]):
                        get_data_eval_2 = torch.tensor([get_data_eval_2.item()], dtype=torch.float32)
                    # Append
                    try:
                        all_data_eval = torch.cat((get_data_eval_1, get_data_eval_2))
                    except:
                        raise AssertionError('Cannot append the torch tensors')
                    if DEBUG:
                        print('Appended Data : ', all_data_eval)
                    return [all_data_eval, sig]
                else:
                    # ['First'/'last'/'rest', ['vector', 2, 3, 4, 5]]
                    e1 = tail
                    if isinstance(e1, list):
                        get_data_struct, sig = evaluate_program(e1, sig=sig, l=l)
                    else:
                        # Most likely a pre-defined varibale in l
                        get_data_struct = l[e1]
                    if DEBUG:
                        print('Data : ', get_data_struct)
                    return [op_func(get_data_struct), sig]

            # Assign
            elif root == 'let':
                # (let [params] body)
                let_param_name  = tail[0][0]
                let_param_value = tail[0][1]
                let_body = tail[1]
                if DEBUG:
                    print('Let param name: ', let_param_name)
                    print('Let params value: ', let_param_value)
                    print('Let body: ', let_body)
                # Evaluate params
                let_param_value_eval, sig = evaluate_program([let_param_value], sig=sig, l=l)
                # Add to local variables
                l[let_param_name] = let_param_value_eval
                # Check for single instance string
                if isinstance(let_body, str):
                    let_body = [let_body]
                if DEBUG:
                    print('Local Params :  ', l)
                    print('Recursive Body: ', let_body, "\n")
                # Evaluate body
                return evaluate_program([let_body], sig=sig, l=l)

            # Conditonal
            elif root == "if":
                # (if e1 e2 e3)
                if DEBUG:
                    print('Conditonal Expr1 :  ', tail[0])
                    print('Conditonal Expr2 :  ', tail[1])
                    print('Conditonal Expr3 :  ', tail[2])
                e1_, sig = evaluate_program([tail[0]], sig, l=l)
                if DEBUG:
                    print('Conditonal eval :  ', e1_)
                if e1_:
                    return evaluate_program([tail[1]], sig, l=l)
                else:
                    return evaluate_program([tail[2]], sig, l=l)

            # Conditional Evaluation
            elif root in cond_ops.keys():
                # (< a b)
                op_func = cond_ops[root]
                if DEBUG:
                    print('Conditional param-1: ', tail[0])
                    print('Conditional param-2: ', tail[1])
                a, sig = evaluate_program([tail[0]], sig, l=l)
                b, sig = evaluate_program([tail[1]], sig, l=l)
                # If torch tensors convert to python data types for comparison
                if torch.is_tensor(a):
                    a = a.tolist()
                    if isinstance(a, list):
                        a = a[0]
                if torch.is_tensor(b):
                    b = b.tolist()
                    if isinstance(b, list):
                        b = b[0]
                if DEBUG:
                    print('Eval Conditional param-1: ', a)
                    print('Eval Conditional param-2: ', b)
                return [op_func(a, b), sig]

            # Functions
            elif root == "defn":
                # (defn name[param] body, )
                if DEBUG:
                    print('Defn Tail: ', tail)
                try:
                    fnname   = tail[0]
                    fnparams = tail[1]
                    fnbody   = tail[2]
                except:
                    raise AssertionError('Failed to define function!')
                if DEBUG:
                    print('Function Name : ', fnname)
                    print('Function Param: ', fnparams)
                    print('Function Body : ', fnbody)
                # Check if already present
                if fnname in rho.keys():
                    return [fnname, sig]
                else:
                    # Define functions
                    rho[fnname] = [fnparams, fnbody]
                    if DEBUG:
                        print('Local Params : ', l)
                        print('Global Funcs : ', rho, "\n")
                    return [fnname, sig]

            # Most likely a single element list or function name
            else:
                if DEBUG:
                    print('End case Root Value: ', root)
                    print('End case Tail Value: ', tail)
                # Check in local vars
                if root in l.keys():
                    return [l[root], sig]
                # Check in Functions vars
                elif root in rho.keys():
                    # import pdb; pdb.set_trace()
                    fnparams_ = {**l}
                    fnparams, fnbody =rho[root]
                    if len(tail) != len(fnparams):
                        raise AssertionError('Function params mis-match!')
                    else:
                        for k in range(len(tail)):
                            fnparams_[fnparams[k]] = evaluate_program([tail[k]], sig=sig, l=l)[0]
                    if DEBUG:
                        print('Function Params :', fnparams_)
                        print('Function Body :', fnbody)
                    # Evalute function body
                    eval_output, sig = evaluate_program([fnbody], sig=sig, l=fnparams_)
                    if DEBUG:
                        print('Function evaluation output: ', eval_output)
                    return [eval_output, sig]
                else:
                    return [root, sig]
        except:
            # Just a single element
            return [e, sig]
    else:
        # Parse functions
        for func in range(len(e)-1):
            if DEBUG:
                print('Function: ', e[func])
            fname, _ = evaluate_program([e[func]], sig=sig, l=l)
            if DEBUG:
                print('Parsed function: ', fname)
                print("\n")
        # Evaluate Expression
        try:
            outputs_, sig = evaluate_program([e[-1]], sig=sig, l=l)
        except:
            raise AssertionError('Failed to evaluate expression!')
        if DEBUG:
            print('Final output: ', outputs_)
        # Return
        return [outputs_, sig]

    return [None, sig]


## Black Box Variational Inference
def optimizer_step(q, g_):
    for v in g_.keys():
        lambda_v  = Q[v].Parameters()
        lambda_v_ = lambda_v + SGD
        Q[v].Parameters()

    return Q_


def elbo_gradients(G_1toL , logW_1toL):
    G_all = []
    for i in range(G_1toL):
        G_all.extend(G_1toL[i])

    F  = {}
    g_ = {}
    for v in G_all:
        for l in range(len(G_1toL)):
            if v in G_1toL[l].keys():
                if l in F.keys():
                    F[l][v] = G_1toL[v] * logW_1toL
                else:
                    F[l] = {}
                    F[l][v] = 0.0
                    G_1toL[v] = 0.0
        b_  = sum(covar(), G_1toL[v])/sum()
        g_v = sum()

    return g_


def BBVI(S, L, T):
    sigma = {}
    sigma["logW"] = 0.0
    sigma["g"] = []
    sigma["G"] = []
    outputs = []
    for t in range(T):
        r_tl = []
        logW_tl = []
        for l in range(L):
            r_tl, sigma_tl = eval(e, sig, l)
            G_tl.append(sigma_tl["G"])
            logW_tl.append(sigma_tl["logW"])
        g_ = elbo_gradients(G_1toL=G_tl , logW_1toL=logW_tl)
        sigma[Q] = optimizer_step(q=sigma[Q], g_=g_)
        # Collect
        outputs.append([r_tl, logW_tl[L-1]])

    return outputs


if __name__ == '__main__':
    # Change the path
    program_path = '/home/tonyjo/Documents/prob-prog/CS539-HW-4'

    for i in range(5,6):
        ## Note: this path should be with respect to the daphne path!
        # ast = daphne(['graph', '-i', f'{program_path}/src/programs/{i}.daphne'])
        # ast_path = f'./jsons/graphs/final/{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)
        # print('\n\n\nSample of prior of program {}:'.format(i))

        if i == 1:
            print('Running evaluation-based-sampling for Task number {}:'.format(str(i)))
            ast_path = f'./jsons/HW3/eval/{i}.json'
            with open(ast_path) as json_file:
                ast = json.load(json_file)
