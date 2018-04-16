
def gini_impurity(probs):
    x = []
    for p in probs:
        x.append(p*(1-p))
    return sum(x)



# splitting on Eye Colour yields
# [Brown, Blue, Hazel, Green]
# Not so good
probclass = [0.35, 0.25, 0.15, 0.25 ]
gini_impurity(probclass)

# splitting on Gender Yields
# [Male, Female]
# Best
probclass = [0.67, 0.33 ]
gini_impurity(probclass)


print('Better to split on Gender')


# splitting on Postcode Yields
# Middle
probclass = [0.1, 0.15, 0.05, 0.02, 0.6, 0.08 ]
gini_impurity(probclass)