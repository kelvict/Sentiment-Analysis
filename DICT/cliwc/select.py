

# neg_words = [x.strip() for x in open('neg').readlines()]

# swear_words =  [x.strip() for x in open('swear').readlines()]

# words = list(set(neg_words + swear_words))
# with open('negative', 'w') as xs:
#     for w in words:
#         if '*' in w or len(w.decode('utf8'))==1:
#             print w.decode('utf8')
#             continue
#         else:
#             xs.write(w+'\t'+'-1'+'\n')




words = [x.strip() for x in open('pos').readlines()]

with open('positive', 'w') as xs:
    for w in words:
        if '*' in w or len(w.decode('utf8'))==1:
            print w.decode('utf8')
            continue
        else:
            xs.write(w+'\t'+'1'+'\n')
