names = {}
cards_ = [[] for i in range(56)]

for line in open('symbols.txt', 'r'):
    dirname, filename = line[:-1].split('\\')
    mark = int(dirname[:2])
    name = dirname[2:]
    names.update({mark:name})
    prop, dense, b, g, cardname = filename.split('_')
    cardnumber = int(cardname[4:cardname.find('.')])
    if mark < 57:
        cards_[cardnumber].append(mark)

cards = {}
for i in range(1, 56):
    cards.update({i:sorted(cards_[i])})
    print('Card', i, end=': ')
    for n in cards[i]:
        print(names[n], end = ', ')
    print()

print('names=', names)
print('cards=', cards)
