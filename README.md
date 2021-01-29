# SpotIt
Slice SpotIt cards to separate symbols and train neural net on it.

We need card pictures near 1000x1000 pixels, named card1.jpg .. card55.jpg.
1. Run slice_auto.py to slice separate symbols and put it into same dirs. Names contain information of geometry and color properties.
(Update 29.01.2021: slise_auto.py uses clustering from sklearn to group files.)
2. Move symbols to right dirs, named like '00Anchor', '01Apple' etc.
3. Get list of files (for example, 'dir /s /b /a-d') and save it with name 'symbols.txt'
4. Run train_symbols.py to train neural net. The net has ben created by example of book Python Machine Learning, lesson 12 https://github.com/rasbt/python-machine-learning-book-3rd-edition/blob/master/ch12/ch12.py
5. Run get_dicts.py to show information about cards. Edit 'names' and 'cards' in show_cards.py
6. Run show_cards.py to check cards
7. Run score.py to measure time for all 1485 combinations of two cards
