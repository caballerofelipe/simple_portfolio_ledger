



# This project should answer these questions

- What was the return of what was invested?
  - Separated by instrument.
  - Including costs, dividends, and everything else.
- What was the price payed for a given instrument?
  - When sold, the price to consider could be: average, FIFO, LIFO. To be determined. It will probably be the average.
   - Maybe the more logical approach is using the average.
   - But, the average should only consider what was bought, not something deposited, given or stock dividend. To be determined
- If an instrument was not bought (e.g. deposited or dividend):
  - ... This shouldn't have a return.
  - The logic is the following: If I was gifted USD 100 and I sell them, the profit is the totality of the sell but the return would be infinite.
  - This also creates a philosophical question:
    - Everything we are and everything with which we start was a gift.
    - So everything we have has given us an infinite return since our initial investment was 0.
    - This also poses another philosophical question: Did we have any incidence over where we begun in the world? If that's the case, I don't remember.
- Does it make sense that if something was given to me, the price I payed for the gift was 0?
  - If you consider 0 for an instrument's price, the average price would come down and wouldn't represent the market price.
- If an instrument returns a dividend: Does it make sense to consider the dividend in the price of what I bought? (I believe not)
- Should the price of something be recalculated only when there's a buy?




# Some ideas about the library



## Every dollar is equal
- This applies to every instrument. They are fungible.
- To calculate profit: it's only important to know how much of a instrument was sold with a profit



## invest / uninvest
...



## Withdrawing and selling an instrument
- If an instrument is sold, the order from which the money is taken out is the following: [deposit, stock dividend, dividend, buy] or maybe specify when doing a `operation_columns_balance`.
- This has the implication that when selling, depending on where the money is taken from, there is or isn't a profit/loss associated with the transaction.
- For instance, if I have an instrument that I deposited and that I bought, let's say I have 100 shares of XYZ: 60 bough (at 10) and 40 deposited. If I sell 60 shares at 20 and I sell the deposited ones first, those deposited (40) shouldn't be accounted as profit since I deposited then, but the remaining 20 would have a 10 profit per share. If it's the other way around, I sell all 60 shares from the ones I bought, I would get a 10 profit for all 60 shares.

# IDEAS
- Make configurable the order from where the money is taken
  - The order from where the instruments are taking could be changed and that would have profit implications.