import numpy as np
import operator
from itertools import combinations
import gurobipy as grb
import timeit
from itertools import chain



k = 2
print("k = ", k)

coef = 0.5
print("lambda coefficient =", coef)

delta = 0.1
print("delta=", delta)

error = 0.0005






sorted_rel = {}
instance = ["i1","i2","i3","i4"]
sorted_rel["i3"] = 3
sorted_rel["i2"] = 2
sorted_rel["i1"] = 2
sorted_rel["i4"] = 1

max_rel = list(sorted_rel.values())[0]
min_rel = list(sorted_rel.values())[len(sorted_rel)-1]

print("Maximum Relevance score: ", max_rel)
print("Minimum Relevance score: ", min_rel)



pairwise_diversity = {}
pairwise_diversity[("i1","i3")] = 10
pairwise_diversity[("i1","i4")] = 5
pairwise_diversity[("i2","i3")] = 5
pairwise_diversity[("i2","i4")] = 5
pairwise_diversity[("i3","i4")] = 5
pairwise_diversity[("i1","i2")] = 4







#####leximin

def leximin(panel_items):

    m = grb.Model()
    # Variables for the output probabilities of the different panels
    numSet = len(panel_items)

    itemSet = set(list(chain.from_iterable([list(itm) for itm in panel_items])))
    numItems = len(list( itemSet))
    lambda_p = m.addVars(numSet,vtype=grb.GRB.CONTINUOUS, lb=0.,name='lambda_p')
    # To avoid numerical problems, we formally minimize the largest downward deviation from the fixed probabilities.
    x = m.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.,name='x')

    item_probability = m.addVars(numItems,vtype=grb.GRB.CONTINUOUS, lb=0.,name='item_probability')


    m.addConstr(grb.quicksum(lambda_p) == 1)  # Probabilities add up to 1

    #item_probability = m.addVar(vtype=grb.GRB.CONTINUOUS, lb=0.)
    i = 0

    vals = []
    for item in itemSet:
        j = 0
        for pi in panel_items:
            if item in list(pi):
                vals.append(lambda_p[j])
            j = j + 1

        # vals = [comm_var for committee, comm_var in zip(panel_items, lambda_p)
        #                                    if item in list(committee)]
        item_probability[i] = grb.quicksum(vals)
        m.addConstr(item_probability[i] >= x)
        i = i + 1
    m.setObjective(x, grb.GRB.MAXIMIZE)
    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))


    probabilities = np.array([comm_var.x for comm_var in lambda_p]).clip(0, 1)
    probabilities = list(probabilities / sum(probabilities))

    finalsetprobs = {}

    for i in range(len(panel_items)):
        for p in range(len(probabilities)):
            if i == p:
                finalsetprobs[panel_items[i]] = probabilities[p]

    print("final panels probabilities: ", finalsetprobs)

    nonzero_prob = {}
    for k, v in finalsetprobs.items():
        if v != 0:
            nonzero_prob[k] = v

    print("non zero panels probabilities:", nonzero_prob)

    print("Size of non zero probability list:", len(nonzero_prob))

    prob = 0
    for i, j in nonzero_prob.items():
        prob = prob + j

    print("total prob:", prob)

    item_probs = {}
    for i in instance:
        p = 0
        for item, k in nonzero_prob.items():
            if i in item:
                p = p + k

        item_probs[i] = p

    print("item probs:", item_probs)
    print("Minimum probability of items: ", min(list(item_probs.values())))

    for v in m.getVars():
        print(v.varName, v.x)

    return nonzero_prob, item_probs

#######################


sorted_pairdiv = dict( sorted(pairwise_diversity.items(), key=operator.itemgetter(1),reverse=True))

print("sorted pairwise diversity: ", sorted_pairdiv)

max_div = list(sorted_pairdiv.values())[0]
min_div = list(sorted_pairdiv.values())[len(sorted_pairdiv)-1]

print("Maximum Diversity score: ", max_div)
print("Minimum Diversity score: ", min_div)


columns = {}
panel = []
topkSets = {}
seen_relist = []
seen_divpairlist = []



minrel = float('inf')

for i in instance:
    rel = sorted_rel[i]
    if rel < minrel:
        minrel = rel

print("Least relevant item in the top-k set:", list(sorted_rel.keys())[list(sorted_rel.values()).index(minrel)])



def getNextRel(position):
    return (list(sorted_rel.keys())[position],list(sorted_rel.values())[position])

#print(getNextRel(0))

def getNextDiv(position):
    return (list(sorted_pairdiv.keys())[position],list(sorted_pairdiv.values())[position])

##############################

def exactMMR(s):
    s = sorted(s)
    s = tuple(s)
    MMR_score = 0
    for item in s:

        maxsim = 0
        for elems in s:
            if item != elems:
                pairs = [item, elems]
                pairs.sort()
                tup = tuple(pairs)
                sim = sorted_pairdiv[tup]
                if maxsim < sim:
                    maxsim = sim
        MMR_i = coef * sorted_rel[item] + (1 - coef) * (maxsim)
        MMR_score = MMR_i + MMR_score

    return MMR_score


 ############################

topkSets = {}
topkSets[0] = float('inf')
init_j = 1
i = 2

threshold = []
Lbounds = {}
Ubounds = {}

seen_items = []
seen_rellist = []
seen_divpairlist = []
########################### NRA algorithm

def generateTopkSet(i, lastMax):
    global init_j
    global seen_items
    global seen_rellist
    global seen_divpairlist


    for j in range(init_j,len(sorted_rel)):



        nextRel_score = getNextRel(j-1)[1]
        nextDiv_score = getNextDiv(j-1)[1]

        nextRel_item = getNextRel(j-1)[0]
        nextDiv_items = getNextDiv(j-1)[0]

        seen_items.append(nextRel_item)
        seen_rellist.append(nextRel_item)

        seen_divpairlist.append(nextDiv_items)

        # items = []
        thisiteritems = []

        thisiteritems.append(nextRel_item)

        for elem in nextDiv_items:
            seen_items.append(elem)
            thisiteritems.append(elem)


        seen_items = list(set(seen_items))

        thisiteritems = set(thisiteritems)
        thisiteritems = list(thisiteritems)

        if len(seen_items) < k:
            continue
        else:

            ###################### threshold and upperbound calculation

            start = timeit.default_timer()

            candidate_sets = []
            for set1 in combinations(seen_items, k):
                set1 = sorted(set1)
                set1 = tuple(set1)
                if set1 in topkSets.values():
                    continue


                candidate_sets.append(set1)
                # print(i)

                MMR_score = 0
                for item in set1:

                    maxsim = 0
                    for elem in set1:
                        if item != elem:

                            if (item, elem) in seen_divpairlist:
                                sim = sorted_pairdiv[(item, elem)]
                                if maxsim < sim:
                                    maxsim = sim
                            elif (elem, item) in seen_divpairlist:
                                sim = sorted_pairdiv[(elem, item)]
                                if maxsim < sim:
                                    maxsim = sim
                            else:
                                sim = nextDiv_score
                                if maxsim < sim:
                                    maxsim = sim
                                    # get rel and div from dictionary

                    if item in seen_rellist:
                        MMR_i = coef * sorted_rel[item] + (1 - coef) * (maxsim)
                    else:
                        MMR_i = coef * (nextRel_score) + (1 - coef) * (maxsim)

                    MMR_score = MMR_i + MMR_score

                Ubounds[set1] = MMR_score

            all_values = Ubounds.values()
            max_MMR = max(all_values)

            MMR_threshold = max_MMR
            threshold.append(MMR_threshold)
            print("Threshold = ", MMR_threshold)

            # if MMR_threshold < dif:
            #    print("break")
            #    break

            ##########################

            ###################### lower bound

            for sets in combinations(seen_items, k):
                # print(i)

                sets = sorted(sets)
                sets = tuple(sets)
                MMR_score = 0
                if sets in topkSets.values():
                    continue
                for item in sets:

                    maxsim = 0
                    for elem in sets:
                        if item != elem:

                            if (item, elem) in seen_divpairlist:
                                sim = sorted_pairdiv[(item, elem)]
                                if maxsim < sim:
                                    maxsim = sim
                            elif (elem, item) in seen_divpairlist:
                                sim = sorted_pairdiv[(elem, item)]
                                if maxsim < sim:
                                    maxsim = sim
                            else:
                                sim = min_div
                                if maxsim < sim:
                                    maxsim = sim
                                    # get rel and div from dictionary

                    if item in seen_rellist:
                        MMR_i = coef * sorted_rel[item] + (1 - coef) * (maxsim)
                    else:
                        MMR_i = coef * (min_rel) + (1 - coef) * (maxsim)

                    MMR_score = MMR_i + MMR_score

                Lbounds[sets] = MMR_score


            ############### pruning and stopping condition
            if len(Ubounds) > 1:
                for settt in Lbounds.keys():
                    maxupscore = []
                    for set2 in Ubounds.keys():
                        if set2 != settt:
                            maxupscore.append(Ubounds[set2])
                    maxup = max(maxupscore)
                    if Lbounds[settt] > maxup:
                        init_j = j + 1
                        print(init_j)
                        return settt  # we found the next best set


            ############################ pruning condition
            if len(Ubounds) > 1:
                for setlb in Lbounds.keys():
                    maxlbscore = []
                    for setlb2 in Lbounds.keys():
                        if setlb2 != setlb:
                            maxlbscore.append(Lbounds[setlb2])
                    maxlb = max(maxlbscore)

                    if Ubounds[setlb] < maxlb:
                        candidate_sets.remove(setlb)  # only for this iteration this set gets pruned


            # stopping condition
            if i > 1:
                if max(Lbounds.values()) >= min(threshold[i], lastMax):
                    init_j = j + 1

                    mmr = 0
                    for eachset in candidate_sets:
                        mmrset = exactMMR(eachset)
                        if mmr < mmrset:
                            mmr = mmrset
                            best = eachset
                    return best



###########################################
#### main


topkSets[1] = generateTopkSet(1,minrel)
theta = Ubounds[topkSets[1]]-delta


i =2
topkSets.pop(0)
while Ubounds[topkSets[i-1]] >= theta:
    lastMax = Ubounds[topkSets[i - 1]]
    Ubounds.pop(topkSets[i-1])
    Lbounds.pop(topkSets[i-1])
    topkSets[i] = generateTopkSet(i,lastMax)



    set_prob, item_prob = leximin(list(topkSets.values()))

    i = i+1

print(set_prob, item_prob)
print(init_j)
