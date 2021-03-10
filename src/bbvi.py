import json
import math
import copy
import numpy as np
import seaborn as sns
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
#torch.autograd.set_detect_anomaly(True)
# OPS
basic_ops = {'+':torch.add,
             '-':torch.sub,
             '*':torch.mul,
             '/':torch.div
}

one_ops = {'sqrt': lambda x: _squareroot(x),
           'vector': lambda x: _vector(x),
           'hash-map': lambda x: _hashmap(x),
           'abs': lambda x: torch.abs(x),
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
            "uniform-continuous": lambda low, high: distributions.uniform.Uniform(low=low, high=high),
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
        print('\n')

    if root == "sample*":
        sample_eval = ["sample", tail]
        if DEBUG:
            print('Sample AST: ', sample_eval)
        sampler, sig = evaluate_program(e=[sample_eval], sig={**sig}, l={**l})
        try:
            p = sampler.sample()
        except:
            # For some reason if it is not a sampler object
            raise AssertionError('Failed to sample!')

        # if v not in (sig["Q"]).keys():
        #     sig["Q"][v] = sampler
        # import pdb; pdb.set_trace()
        try:
            sig_Q_v_new = (sig["Q"][node]).make_copy_with_grads(init=False)
            try:
                c = sig_Q_v_new.sample()
            except:
                # For some reason if it is not a sampler object
                raise AssertionError('Failed to sample!')
            if DEBUG:
                print('Current Root:', node)
                print('Current Sig: ', sig_Q_v_new)
                print('Current Sample C: ', c)
                print('\n')
            nlp = -sig_Q_v_new.log_prob(c)
            nlp.backward()
            grad_GD = torch.zeros(0, dtype=torch.float32)
            for param in sig_Q_v_new.Parameters():
                grad_gd = param.grad.clone()
                # Check if not torch tensor
                if not torch.is_tensor(grad_gd):
                    if isinstance(grad_gd, list):
                        grad_gd = torch.tensor(grad_gd, dtype=torch.float32)
                    else:
                        grad_gd = torch.tensor([grad_gd], dtype=torch.float32)
                # Check for 0 dimensional tensor
                elif grad_gd.shape == torch.Size([]):
                    grad_gd = torch.tensor([grad_gd.item()], dtype=torch.float32)
                # Concat
                try:
                    grad_GD = torch.cat((grad_GD, grad_gd))
                except:
                    raise AssertionError('Cannot append the torch tensors')
            if DEBUG:
                print("grad_GD: ", grad_GD)
                print('\n')
            sig["G"][node] = grad_GD.clone().detach()
        except:
            # Its not a needed Q variable
            return [c, sig]

        logWv = sampler.log_prob(c) - (sig_Q_v_new).log_prob(c)
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
        c, sig = evaluate_program(e=[sample_eval], sig={**sig}, l={**l})

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
DEBUG = False # Set to true to see intermediate outputs for debugging purposes
def eval_graph(graph, sigma={}, l={}):
    """
    This function evaluate the graph.
    Args:
        graph: json Graph of FOPPL program
    """
    D, G, E = graph[0], graph[1], graph[2]

    # Compiled graph
    V = G['V']
    A = G['A']
    P = G['P']
    Y = G['Y']

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
                eval_outputs.append([output_])
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
def optimizer_step(g_, Q, lr=0.0001):
    for v in g_.keys():
        q_v = Q[v]
        old_params = q_v.Parameters()
        grad_Lv = g_[v]
        if DEBUG:
            print("old_params: ", old_params)
            print("grad_Lv: ", grad_Lv)
        # SGD
        new_params = old_params + (lr * grad_Lv)
        for idx in range(len(new_params)):
            new_params[idx] = new_params[idx].detach().clone()
        if DEBUG:
            print("New Q Params: ", new_params)
        # Update
        Q[v] = Q[v].Set_Parameters(new_params)

    return Q


def elbo_gradients(G_1toL , logW_1toL, Q, L, eps=0.00001):
    # import pdb; pdb.set_trace()
    G_all = []
    for i in range(L):
        G_all.extend(G_1toL[i])
    G_all = set(G_all)
    if DEBUG:
        print("G_1toL: ", G_1toL)
        print("logW1L: ", logW_1toL)
        print("G_all: ", G_all)

    F  = {}
    g_ = {}
    for v in G_all:
        for l in range(L):
            if v in G_1toL[l]:
                if l not in F.keys():
                    F[l] = {}
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
            g1L = G_1toL[l][v].detach()
            g1L.requires_grad = False
            g1L = g1L.cpu().numpy()
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

        #Flv = Flv.flatten('F')
        #G1L = G1L.flatten('F')
        if DEBUG:
            print("Flv: \n", Flv)
            print("G1L: \n", G1L)
            print("Conv: \n", np.cov(Flv, G1L))
            print('Sum_GIL: ', np.sum(G1L))
            print('\n')

        num = np.sum(np.cov(Flv.T, G1L.T))
        den = np.sum(np.var(G1L))
        if den == 0. or den != den: # catch nan
            b_ = 0.
        elif abs(den) < 0.0001:
            den = 0.0001
        else:
            b_ = num/den

        g_[v] = np.sum(Flv - (b_ * G1L))/L

    return g_

def BBVI(graph, Q, S, L, T, display_T=10):
    sigma = {}
    sigma["logW"] = 0.0
    sigma["q"] = {}
    sigma["G"] = {}
    sigma["Q"] = {**Q}

    outputs = []
    for t in range(T):
        R_tL = []
        G_tL = []
        logW_tL = []
        for l in range(L):
            # import pdb; pdb.set_trace()
            r_tl, sigma_tl = eval_graph(graph=graph, sigma=sigma, l={})
            G_tL.append({**sigma_tl["G"]})
            logW_tL.append(sigma_tl["logW"])
            R_tL.append([r_tl, sigma_tl["logW"]])
        # import pdb; pdb.set_trace()
        g_ = elbo_gradients(G_1toL=G_tL, logW_1toL=logW_tL, Q={**Q}, L=L)
        Q  = optimizer_step(g_={**g_}, Q=Q)
        # Update Q
        sigma["Q"] = {**Q}
        # Collect
        outputs.extend(R_tL)
        # Display
        if t%display_T == 0:
            print(f'Completion: {t}/{T}')
    print(f'Completion: {T}/{T}')

    return outputs


if __name__ == '__main__':
    # Change the path
    program_path = '/home/tonyjo/Documents/prob-prog/CS539-HW-4'

    #for i in range(5,6):
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
            print('Running BBVI for Task number {}:'.format(str(i)))
            graph_path = f'./jsons/graphs/final/{i}.json'
            with open(graph_path) as json_file:
                graph = json.load(json_file)

            # Setup proposal -- same as the prior distribution
            Q = {}
            loc   = torch.tensor(0.0)
            scale = torch.tensor(5.0)
            V = graph[1]["V"]
            for v in V:
                Q[v] = Normal(loc, scale).make_copy_with_grads()

            # Setup up
            # Print
            T  = 1000
            L  = 5
            DT = 100
            outputs = BBVI(graph=graph, Q=Q, S=1, L=L, T=T, display_T=DT)

            # Mean:
            W_k = 0.0
            for k in range(T):
                r_l, W_l = outputs[k]
                W_l = W_l.item()
                W_k += math.exp(W_l)

            EX = 0.0
            for l in range(T):
                r_l, W_l = outputs[l]
                r_l = r_l.item()
                W_l = W_l.item()
                W_l = math.exp(W_l)
                EX += ((W_l/W_k) * r_l)
            print("Posterior Mean: ", EX)
            print("--------------------------------")
            print("\n")

            EX2 = 0.0
            for l in range(T):
                r_l, W_l = outputs[l]
                r_l = r_l.item()
                W_l = W_l.item()
                W_l = math.exp(W_l)
                EX2 += ((W_l/W_k) * (r_l**2))
            var = EX2 - (EX**2)
            print("Posterior Variance:", var)
            print("--------------------------------")
            print("\n")

            plt.hist([r[0].item() for r in outputs])
            plt.savefig('plots/1.png')
            plt.clf()

            jp = []
            for r in range(0, len(outputs), L):
                joint_log_prob = 0.0
                for l in range(L):
                    x1 = outputs[r+l]
                    if isinstance(x1, list):
                        x1 = x1[0]
                    try:
                        joint_log_prob += -x1
                    except:
                        joint_log_prob += -x1[0]
                jp.append(joint_log_prob)

            plt.plot(jp)
            plt.savefig('plots/1_1.png')
            plt.clf()
        #-----------------------------------------------------------------------
        elif i == 2:
            print('Running graph-based-sampling for Task number {}:'.format(str(i)))
            graph_path = f'./jsons/graphs/final/{i}.json'
            with open(graph_path) as json_file:
                graph = json.load(json_file)

            # Setup proposal -- same as the prior distribution
            Q = {}
            loc   = torch.tensor([0.])
            scale = torch.tensor([1.])
            V = graph[1]["V"]
            for v in V:
                if v in 'sample1':
                    loc_1   = torch.tensor([0.])
                    scale_1 = torch.tensor([10.])
                    Q[v] = Normal(loc_1, scale_1).make_copy_with_grads()
                elif v in 'sample2':
                    loc_2   = torch.tensor([0.])
                    scale_2 = torch.tensor([10.])
                    Q[v] = Normal(loc_2, scale_2).make_copy_with_grads()
                else:
                    Q[v] = Normal(loc, scale).make_copy_with_grads()

            # Setup up
            # Print
            T  = 10000
            L  = 5
            DT = 100
            all_output = BBVI(graph=graph, Q=Q, S=1, L=L, T=T, display_T=DT)

            W_k = 0.0
            for k in range(T):
                r_l, W_l = all_output[k]
                W_l = W_l.item()
                W_k += math.exp(W_l)

            EX_slope = 0.0
            EX_bias = 0.0
            for l in range(T):
                r_l, W_l = all_output[l]
                W_l = W_l.item()
                W_l = math.exp(W_l)
                EX_slope += ((W_l/W_k) * r_l[0].item())
                EX_bias  += ((W_l/W_k) * r_l[1].item())
            print("Posterior Bias  Mean: ", EX_bias)
            print("Posterior Slope Mean: ", EX_slope)
            print("--------------------------------")
            print("\n")

            EX2_ = []
            for l in range(T):
                r_l, W_l = all_output[l]
                W_l = W_l.item()
                W_l = math.exp(W_l)
                EX2_.extend([(W_l/W_k) * r_l[0].item() * r_l[1].item()])
            covar = sum(EX2_) - (EX_slope * EX_bias)
            print("Posterior Covariance : ", covar)
            print("---------------------------------")
            print("\n")
        #-----------------------------------------------------------------------
        elif i == 3:
            print('Running graph-based-sampling for Task number {}:'.format(str(i)))
            graph_path = f'./jsons/graphs/final/{i}.json'
            with open(graph_path) as json_file:
                graph = json.load(json_file)

            # Setup proposal -- same as the prior distribution
            Q = {}
            V = graph[1]["V"]
            P = graph[1]["P"]
            d_ops = {"normal":lambda mu, sig: Normal(loc=mu, scale=sig),
                     "bernoulli": lambda probs: Bernoulli(probs=probs),
                     "categorical": lambda probs: Categorical(probs=probs),
                     "discrete": lambda probs: Categorical(probs=probs),
                     "dirichlet": lambda concentration: Dirichlet(concentration=concentration),
                     "gamma": lambda concentration, rate: Gamma(concentration=concentration, rate=rate)
            }
            peval = d_ops["dirichlet"]
            probs = torch.tensor([1.0, 1.0, 1.0])
            Q["sample6"] = peval(probs)
            loc   = torch.tensor([0.])
            scale = torch.tensor([1.])
            for v in V:
                if ("sample" in v) and (v != "sample6"):
                    try:
                        p = P[v][1]
                        a, b, c = p
                        b = torch.tensor([b], dtype=torch.float32)
                        c = torch.tensor([c], dtype=torch.float32)
                        peval = d_ops[a]
                        Q[v]  = peval(b, c)
                    except:
                        a, b = p
                        b = Q["sample6"].sample()
                        peval = d_ops[a]
                        Q[v]  = peval(b)
                else:
                    Q[v] = Normal(loc, scale)

            # Setup up
            # Print
            T  = 1000
            L  = 5
            DT = 100
            all_output = BBVI(graph=graph, Q=Q, S=1, L=L, T=T, display_T=DT)

            EX = 0.0
            for l in range(T):
                r_l, W_l = all_output[l]
                r_l = float(r_l)
                EX += r_l
            print("Posterior Mean: ", EX/T)
            print("--------------------------------")
            print("\n")

            EX2 = 0.0
            for l in range(T):
                r_l, W_l = all_output[l]
                r_l = float(r_l)
                EX2 += (float(r_l)**2)
            EX2 = EX2/T
            var = EX2 - (EX**2)
            print("Posterior Variance:", var)
            print("--------------------------------")
            print("\n")
        #-----------------------------------------------------------------------
        elif i == 4:
            print('Running graph-based-sampling for Task number {}:'.format(str(i)))
            graph_path = f'./jsons/graphs/final/{i}.json'
            with open(graph_path) as json_file:
                graph = json.load(json_file)

            # Setup proposal -- same as the prior distribution
            Q = {}
            V = graph[1]["V"]
            P = graph[1]["P"]
            d_ops = {"normal":lambda mu, sig: Normal(loc=mu, scale=sig),
                     "bernoulli": lambda probs: Bernoulli(probs=probs),
                     "categorical": lambda probs: Categorical(probs=probs),
                     "discrete": lambda probs: Categorical(probs=probs),
                     "dirichlet": lambda concentration: Dirichlet(concentration=concentration),
                     "gamma": lambda concentration, rate: Gamma(concentration=concentration, rate=rate)
            }
            loc   = torch.tensor([0.])
            scale = torch.tensor([1.])
            for v in V:
                if ("sample" in v):
                    p = P[v][1]
                    a, b, c = p
                    b = torch.tensor([b], dtype=torch.float32)
                    c = torch.tensor([c], dtype=torch.float32)
                    peval = d_ops[a]
                    Q[v]  = peval(b, c)
                else:
                    Q[v] = Normal(loc, scale)

            # Setup up
            # Print
            T  = 1000
            L  = 5
            DT = 100
            all_output = BBVI(graph=graph, Q=Q, S=1, L=L, T=T, display_T=DT)

            samples = []
            for l in range(T):
                r_l, W_l = all_output[l]
                W_0 = r_l[0:10]
                b_0 = r_l[10:20]
                W_1 = r_l[20:120]
                b_1 = r_l[120:]
                samples.append([W_0, b_0, W_1, b_1])

            # Plotting obtained from https://github.com/truebluejason/prob_prog_project/blob/master/evaluation_based_sampling.py
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2))
            mean1 = [np.mean([s[0].flatten()[j] for s in samples]) for j in range(len(samples[0][0].flatten()))]
            var1  = [np.var([s[0].flatten()[j] for s in samples]) for j in range(len(samples[0][0].flatten()))]
            sns.heatmap(np.array(mean1).reshape(2,5),ax=ax1,annot=True,fmt="0.3f")
            sns.heatmap(np.array(var1).reshape(2,5),ax=ax2,annot=True,fmt="0.3f")
            ax1.set_title('Marginal Mean')
            ax2.set_title('Marginal Variance')
            plt.tight_layout()
            plt.savefig(f'plots/p41.png')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2))
            mean2 = [np.mean([s[1].flatten()[j] for s in samples]) for j in range(len(samples[0][1].flatten()))]
            var2  = [np.var([s[1].flatten()[j] for s in samples]) for j in range(len(samples[0][1].flatten()))]
            sns.heatmap(np.array(mean2).reshape(2,5),ax=ax1,annot=True,fmt="0.3f")
            sns.heatmap(np.array(var2).reshape(2,5),ax=ax2,annot=True,fmt="0.3f")
            ax1.set_title('Marginal Mean')
            ax2.set_title('Marginal Variance')
            plt.tight_layout()
            plt.savefig(f'plots/p42.png')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,7))
            mean3 = [np.mean([s[2].flatten()[j] for s in samples]) for j in range(len(samples[0][2].flatten()))]
            var3  = [np.var([s[2].flatten()[j] for s in samples]) for j in range(len(samples[0][2].flatten()))]
            sns.heatmap(np.array(mean3).reshape(10,10),ax=ax1,annot=True,fmt="0.3f")
            sns.heatmap(np.array(var3).reshape(10,10),ax=ax2,annot=True,fmt="0.3f")
            ax1.set_title('Marginal Mean')
            ax2.set_title('Marginal Variance')
            plt.tight_layout()
            plt.savefig(f'plots/p43.png')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,2))
            mean4 = [np.mean([s[3].flatten()[j] for s in samples]) for j in range(len(samples[0][3].flatten()))]
            var4  = [np.var([s[3].flatten()[j] for s in samples]) for j in range(len(samples[0][3].flatten()))]
            sns.heatmap(np.array(mean4).reshape(2,5),ax=ax1,annot=True,fmt="0.3f")
            sns.heatmap(np.array(var4).reshape(2,5),ax=ax2,annot=True,fmt="0.3f")
            ax1.set_title('Marginal Mean')
            ax2.set_title('Marginal Variance')
            plt.tight_layout()
            plt.savefig(f'plots/p44.png')
        #-----------------------------------------------------------------------
        elif i == 5:
            print('Running graph-based-sampling for Task number {}:'.format(str(i)))
            graph_path = f'./jsons/graphs/final/{i}.json'
            with open(graph_path) as json_file:
                graph = json.load(json_file)

            # Setup proposal -- same as the prior distribution
            Q = {}
            loc   = torch.tensor([0.0])
            scale = torch.tensor([1.0])
            V = graph[1]["V"]
            for v in V:
                Q[v] = Normal(loc, scale)

            # Setup up
            # Print
            T  = 100
            L  = 2
            DT = 100
            outputs = BBVI(graph=graph, Q=Q, S=1, L=L, T=T, display_T=DT)

            # Mean:
            W_k = 0.0
            for k in range(T):
                r_l, W_l = outputs[k]
                W_l = W_l.item()
                W_k += math.exp(W_l)

            EX = 0.0
            for l in range(T):
                r_l, W_l = outputs[l]
                r_l = r_l.item()
                W_l = W_l.item()
                W_l = math.exp(W_l)
                EX += ((W_l/W_k) * r_l)
            print("Posterior Mean: ", EX)
            print("--------------------------------")
            print("\n")

            EX2 = 0.0
            for l in range(T):
                r_l, W_l = outputs[l]
                r_l = r_l.item()
                W_l = W_l.item()
                W_l = math.exp(W_l)
                EX2 += ((W_l/W_k) * (r_l**2))
            var = EX2 - (EX**2)
            print("Posterior Variance:", var)
            print("--------------------------------")
            print("\n")

            plt.hist([r[0].item() for r in outputs])
            plt.savefig('plots/5.png')
        #-----------------------------------------------------------------------
