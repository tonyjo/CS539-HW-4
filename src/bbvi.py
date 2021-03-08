import json
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as distributions
from daphne import daphne
# funcprimitives
from eval import evaluate_program
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

one_ops = {'sqrt': lambda x: _squareroot(x),
           'vector': lambda x: _vector(x),
           'hash-map': lambda x: _hashmap(x),
           'first': lambda x: x[0],
           'second': lambda x: x[1],
           'last': lambda x: x[-1],
           'rest': lambda x: x[1:],
           "mat-tanh": lambda a: torch.tanh(a),
           "mat-transpose": lambda a: _mat_transpose(a)
}

two_ops = {'get': lambda x, idx: _get(x, idx),
           'append': lambda x, y: _append(x, y),
           'remove': lambda x, idx: _remove(x, idx),
           ">": lambda a, b: a > b,
           "=": lambda a, b: a == b,
           ">=": lambda a, b: a >= b,
           "<=": lambda a, b: a <= b,
           "or": lambda a, b: a or b,
           "and": lambda a, b: a and b,
           "mat-add": lambda a, b: torch.add(a, b),
           "mat-mul": lambda a, b: torch.matmul(a, b)
}

three_ops = {"mat-repmat": lambda a, b, c: _mat_repmat(a, b, c)
}

dist_ops = {"normal":lambda mu, sig: distributions.normal.Normal(loc=mu, scale=sig),
            "beta":lambda a, b: distributions.beta.Beta(concentration1=a, concentration0=b),
            "gamma": lambda concentration, rate: distributions.gamma.Gamma(concentration=concentration, rate=rate),
            "uniform": lambda low, high: distributions.uniform.Uniform(low=low, high=high),
            "exponential":lambda rate: distributions.exponential.Exponential(rate=rate),
            "discrete": lambda probs: distributions.categorical.Categorical(probs=probs),
            "dirichlet": lambda concentration: distributions.dirichlet.Dirichlet(concentration=concentration),
            "bernoulli": lambda probs: distributions.bernoulli.Bernoulli(probs=probs),
            "flip": lambda probs: distributions.bernoulli.Bernoulli(probs=probs)
}

# GRAPH Utils
def make_link(G, node1, node2):
    """
    Create a DAG
    """
    if node1 not in G:
        G[node1] = {}
    (G[node1])[node2] = 1
    if node2 not in G:
        G[node2] = {}
    (G[node2])[node1] = -1

    return G

def eval_vertex(node, sig={}, l={}, Y={}, P={}):
    """
    Evaluate a node
    """
    lf = P[node] # [sample* [n, 5, [sqrt, 5]]]
    root = lf[0]
    tail = lf[1]
    if DEBUG:
        print('PMF for Node: ', lf)
        print('Empty sample Root: ', root)
        print('Empty sample Root: ', tail)

    if root == "sample*":
        sample_eval = ["sample", tail]
        if DEBUG:
            print('Sample AST: ', sample_eval)
        sampler, sig = evaluate_program(e=[sample_eval], sig=sig, l=l)
        try:
            p = sampler.sample()
        except:
            # For some reason if it is not a sampler object
            raise AssertionError('Failed to sample!')

        # if v not in (sig["Q"]).keys():
        #     sig["Q"][v] = sampler
        try:
            c = (sig["Q"][node]).sample()
        except:
            # For some reason if it is not a sampler object
            raise AssertionError('Failed to sample!')

        try:
            if DEBUG:
                print('Current Root:', node)
                print('Current Sig: ', sig["Q"])
            sig_Q_v = sig["Q"][node]
            sig_Q_v = sig_Q_v.make_copy_with_grads()
            sig["G"][node] = sig_Q_v
            if DEBUG:
                print('Current Root:', sig["G"][node])
        except:
            # Its not a needed Q variable
            return [c, sig]

        logWv = sampler.log_prob(p) - (sig["G"][node]).log_prob(c)
        try:
            if "logW" in sig.keys():
                sig["logW"] += logWv
            else:
                sig["logW"]  = logWv
        except:
            if "logW" in sig.keys():
                sig["logW"] += 0.0
            else:
                sig["logW"]  = 0.0

    elif root == "observe*":
        try:
            sample_eval = ["observe", tail, lf[2]]
        except:
            sample_eval = ["observe", tail]
        if DEBUG:
            print('Sample AST: ', sample_eval)
        c, sig = evaluate_program(e=[sample_eval], sig=sig, l=l)

    else:
        raise AssertionError('Unsupported operation!')

    if DEBUG:
        print('Node eval sample output: ', c)
        print('Node eval sigma  output: ', sig)
        print('\n')

    # Check if not torch tensor
    if not torch.is_tensor(c):
        if isinstance(c, list):
            c = torch.tensor(c, dtype=torch.float32)
        else:
            c = torch.tensor([c], dtype=torch.float32)

    return [c, sig]


# Global vars
global rho
rho = {}
DEBUG = False # Set to true to see intermediate outputs for debugging purposes
def eval_graph(graph, sigma={}, l={}):
    """
    This function does ancestral sampling starting from the prior.
    Args:
        graph: json Graph of FOPPL program
    """
    D, G, E = graph[0], graph[1], graph[2]

    # Compiled graph
    V = G['V']
    A = G['A']
    P = G['P']
    Y = G['Y']

    # Find the link nodes aka nodes not in V
    adj_list = []
    for a in A.keys():
        links = A[a]
        for link in links:
            adj_list.append((a, link))
    # if DEBUG:
    #     print("Created Adjacency list: ", adj_list)

    # Create Graph
    G_ = {}
    for (n1, n2) in adj_list:
        G_ = make_link(G=G_, node1=n1, node2=n2)

    # Test
    # E = "observe8"
    # import pdb; pdb.set_trace()

    if DEBUG:
        print("Constructed Graph: ", G_)
        print("Evaluation Expression: ", E)

    ## Setup Local vars
    # Add Y to local vars
    for y in Y.keys():
        l[y] = Y[y]

    # Evaluate and add sample functions to local vars
    all_sigma = {}
    # First pass is for independent vars
    for  _ in range(2):
        for pi in P.keys():
            p = P[pi]
            root = p[0]
            tail = p[1]
            # Check if already in l
            if pi in l.keys():
                output_ = l[pi]
                if torch.is_tensor(output_):
                    continue
            if root == "sample*":
                try:
                    # Evaluate
                    output_, sigma_ = eval_vertex(pi, sig={**sigma}, l={**l}, Y=Y, P=P)
                    if torch.is_tensor(output_):
                        l[pi] = output_
                        all_sigma[pi] = sigma_
                    else:
                        continue
                except:
                    continue

    # Evaluate and add observation functions to local vars
    for pi in P.keys():
        p = P[pi]
        root = p[0]
        tail = p[1]
        # Check if already in l
        if pi in l.keys():
            output_ = l[pi]
            if torch.is_tensor(output_):
                continue
        if root == "observe*":
            # Evaluate
            output_, sigma_ = eval_vertex(pi, sig={**sigma}, l={**l}, Y=Y, P=P)
            if torch.is_tensor(output_):
                l[pi] = output_
                all_sigma[pi] = sigma_
            else:
                continue

    if DEBUG:
        print('All Local vars: ', l)
        print('All Sigma vars: ', all_sigma)
        print('All Sigma vars: ', all_sigma)
        print('\n')

    # import pdb; pdb.set_trace()
    if isinstance(E, str):
        output = l[E]
        sigma_ = all_sigma[E]
        if DEBUG:
            print('Evaluated graph output: ', output)
            print('Evaluated graph Sigma:  ', sigma_)

        return [output, sigma_]


    elif isinstance(E, list):
        # Evalute
        root_expr, *tail = E
        if DEBUG:
            print('Root OP: ', root_expr)
            print('TAIL: ', tail)
            print('Local vars: ', l)
            print('\n')

        eval_outputs = []
        # Conditonal
        if root_expr == 'if':
            # (if e1 e2 e3)
            if DEBUG:
                print('Conditonal Expr1 :  ', tail[0])
                print('Conditonal Expr2 :  ', tail[1])
                print('Conditonal Expr3 :  ', tail[2])
            e1_, sigma_ = evaluate_program([tail[0]], sig={**sigma}, l={**l})
            if DEBUG:
                print('Conditonal eval :  ', e1_)
            if e1_:
                if tail[1] in V:
                    expression = tail[1]
                    output = l[expression]
                    sigma_ = all_sigma[expression]
                    if DEBUG:
                        print('Evaluated graph output: ', output)
                        print('Evaluated graph Sigma:  ', sigma_)
                else:
                    output, sigma_ = evaluate_program([tail[1]], sig={**sigma}, l={**l})
                    if DEBUG:
                        print('Evaluated graph output: ', output)
                        print('Evaluated graph Sigma:  ', sigma_)
            else:
                if tail[2] in V:
                    expression = tail[2]
                    output = l[expression]
                    sigma_ = all_sigma[expression]
                    if DEBUG:
                        print('Evaluated graph output: ', output)
                        print('Evaluated graph Sigma:  ', sigma_)
                else:
                    output, sigma = evaluate_program([tail[1]], sig={**sigma}, l={**l})

            return [output, sigma]

        # Vector
        elif root_expr == "vector":
            # import pdb; pdb.set_trace()
            if DEBUG:
                print('Data Structure data: ', tail)
            acc_logW = 0.0
            final_sigma = {}
            # Eval tails:
            output = torch.zeros(0, dtype=torch.float32)
            for T in range(len(tail)):
                # Check for single referenced string
                if isinstance(tail[T], str):
                    if tail[T] in V:
                        exp = tail[T]
                        output_ = l[exp]
                        sigma_  = all_sigma[exp]
                    else:
                        output_, sigma_ = evaluate_program([tail[T]], sig={**sigma}, l={**l})
                else:
                    output_, sigma_ = evaluate_program([tail[T]], sig={**sigma}, l={**l})
                if DEBUG:
                    print('Evaluated Data Structure data: ', output_)
                # If sample object then take a sample
                try:
                    output_ = output_.sample()
                except:
                    pass
                # Check if not torch tensor
                if not torch.is_tensor(output_):
                    if isinstance(output_, list):
                        output_ = torch.tensor(output_, dtype=torch.float32)
                    else:
                        output_ = torch.tensor([output_], dtype=torch.float32)
                # Check for 0 dimensional tensor
                elif output_.shape == torch.Size([]):
                    output_ = torch.tensor([output_.item()], dtype=torch.float32)
                # Concat
                try:
                    output = torch.cat((output, output_))
                except:
                    raise AssertionError('Cannot append the torch tensors')
                # Accumlate and update sigma
                acc_logW += sigma_["logW"]
                final_sigma = {**final_sigma, **sigma_}
            if DEBUG:
                print('Eval Data Structure data: ', output)

            # Final acc logW
            final_sigma["logW"] = acc_logW

            return [output, final_sigma]

        # Others
        else:
            acc_logW = 0.0
            final_sigma = {}
            for T in range(len(tail)):
                if tail[T] in V:
                    exp = tail[T]
                    output_ = l[exp]
                    sigma_  = all_sigma[exp]
                else:
                    output_, sigma_ = evaluate_program([tail[T]], sig={**sigma}, l={**l})
                # Accumlate and update sigma
                acc_logW += sigma_["logW"]
                final_sigma = {**final_sigma, **sigma_}
                # Collect
                eval_outputs.append([output])
            if DEBUG:
                print('For eval: ', eval_outputs)
            # Evaluate expression
            if root_expr in one_ops.keys():
                op_func = one_ops[root_expr]
                op_eval = op_func(eval_outputs[0])
            elif root_expr in two_ops.keys():
                op_func = two_ops[root_expr]
                if DEBUG:
                    print('Final output: ', eval_outputs[0])
                    print('Final output: ', eval_outputs[1])
                op_eval = op_func(eval_outputs[0], eval_outputs[1])
            else:
                op_func = three_ops[root_expr]
                if DEBUG:
                    print('Final output: ', eval_outputs[0])
                    print('Final output: ', eval_outputs[1])
                    print('Final output: ', eval_outputs[2])
                op_eval = op_func(eval_outputs[0], eval_outputs[1], eval_outputs[2])

            # Final acc logW
            final_sigma["logW"] = acc_logW

            return [op_eval, final_sigma]
    else:
        raise AssertionError('Invalid input of E!')

    return [None, sig]


## Black Box Variational Inference
def optimizer_step(g_, Q, lr=0.001):
    for v in g_.keys():
        q_v = Q[v]
        # now you can make a copy, that has gradients enabled
        # lambda_v  = q_v.make_copy_with_grads()
        old_params = q_v.Parameters()
        grad_Lv = g_[v]
        if DEBUG:
            print("old_params: ", old_params)
            print("grad_Lv: ", grad_Lv)
        new_lambda_v_params = old_params + (lr * grad_Lv)
        if DEBUG:
            print("New Q Params: ", new_lambda_v_params)
        # Update
        Q[v] = Q[v].Set_Parameters(new_lambda_v_params)

    return Q


def elbo_gradients(G_1toL , logW_1toL, Q, L):
    G_all = []
    for i in range(L):
        G_all.extend(G_1toL[i])
    G_all = set(G_all)
    if DEBUG:
        print("G_all: ", G_all)

    F  = {}
    g_ = {}
    for v in G_all:
        for l in range(L):
            if v in G_1toL[l]:
                if l not in F.keys():
                    F[l] = {}
                try:
                    # If sample object
                    F[l][v] = G_1toL[l][v].sample() * logW_1toL[l]
                except:
                    F[l][v] = G_1toL[l][v] * logW_1toL[l]
            else:
                if l not in F.keys():
                    F[l] = {}
                F[l][v] = 0.0
                G_1toL[l] = 0.0
        # Construct Array
        Flv = []
        G1L = []
        for l in range(L):
            flv = F[l][v]
            flv = flv.detach()
            flv.requires_grad = False
            Flv.append(flv.cpu().numpy())
            try:
                # If sample object
                g1L = G_1toL[l][v].sample()
            except:
                g1L = G_1toL[l][v]
            G1L.append(g1L)
        Flv = np.array(Flv)
        if len(Flv.shape) < 2:
            Flv = np.expand_dims(Flv, axis=1)
        G1L = np.array(G1L)
        if len(G1L.shape) < 2:
            G1L = np.expand_dims(G1L, axis=1)

        if DEBUG:
            print("Flv: \n", Flv)
            print("G1L: \n", G1L)
            print("Conv: \n", np.cov(Flv.T, G1L.T))
            print('Sum_GIL: ', np.sum(G1L))
            print('\n')

        b_ = np.sum(np.cov(Flv.T, G1L.T))/np.sum(G1L)
        g_[v] = np.sum(Flv - (b_ * G1L))/L

    return g_

def BBVI(graph, Q, S, L, T):
    outputs = []
    sigma = {}
    sigma["logW"] = 0.0
    sigma["q"] = {}
    sigma["G"] = {}
    sigma["Q"] = {**Q}

    for t in range(T):
        G_tL = []
        logW_tL = []
        for l in range(L):
            r_tl, sigma_tl = eval_graph(graph=graph, sigma={**sigma}, l={})
            G_tL.append(sigma_tl["G"])
            logW_tL.append(sigma_tl["logW"])
        # import pdb; pdb.set_trace()
        # print(G_tL)
        # print(logW_tL)
        # auto_elbo_gradients(G_1toL=G_tL, logW_1toL=logW_tL, Q=Q, L=L)
        g_ = elbo_gradients(G_1toL=G_tL, logW_1toL=logW_tL, Q={**Q}, L=L)
        Q  = optimizer_step(g_=g_, Q={**Q})
        # Collect
        outputs.append([r_tl, logW_tL[L-1]])
        # Display
        if t%10 == 0:
            print(f'Completion: {t}/{T}')
    print(f'Completion: {T}/{T}')

    return outputs


if __name__ == '__main__':
    # Change the path
    program_path = '/home/tonyjo/Documents/prob-prog/CS539-HW-4'

    for i in range(1,6):
        ## Note: this path should be with respect to the daphne path!
        # ast = daphne(['graph', '-i', f'{program_path}/src/programs/{i}.daphne'])
        # ast_path = f'./jsons/graphs/final/{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)
        # print('\n\n\nSample of prior of program {}:'.format(i))

        # # Note: this path should be with respect to the daphne path!
        # ast = daphne(['desugar', '-i', f'{program_path}/src/programs/{i}.daphne'])
        # ast_path = f'./jsons/eval/final/{i}.json'
        # with open(ast_path, 'w') as fout:
        #     json.dump(ast, fout, indent=2)
        # print('\n\n\nSample of posterior of program {}:'.format(i))

        if i == 1:
            print('Running evaluation-based-sampling for Task number {}:'.format(str(i)))
            ast_path = f'./jsons/eval/final/{i}.json'
            with open(ast_path) as json_file:
                ast = json.load(json_file)

            graph_path = f'./jsons/graphs/final/{i}.json'
            with open(graph_path) as json_file:
                graph = json.load(json_file)

            V = graph[1]["V"]
            print(V)

            Q = {}
            loc   = torch.tensor(0.)
            scale = torch.tensor(10.)

            for v in V:
                Q[v] = Normal(loc, scale)

            # Setup up
            # Print
            outputs = BBVI(graph=graph, Q=Q, S=1, L=5, T=100)
