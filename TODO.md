# TODO LIST
2024-10-17

## Critical
- (Not critical per se but is very similar to previous task, will do it right after) Create operation_list, operation_cumsum, operation_balance.
	- Or maybe where ini and end are dates
		- asset_operations(ini, end, account, instrument, price_in)
		- asset_operations_cumsum(ini, end, account, instrument, price_in)
		- asset_balance(ini, end, account, instrument, price_in)
		- Maybe having a function that returns the same as operation_columns but in a different format would be useful to understand the state of the portfolio. This function would show the groups in a more legible way.
		```
		# Create groups by instrument/operation

		groups = self._ledger_df.groupby(['account', 'instrument', 'price_in', 'operation'])

		# For each group, return the size

		# return directly this

		op_size = groups.apply(lambda x: x['size'], include_groups=False)
		# also the apply could return x['size'].sum()
		```
- When retrieving the ledger, a sorting should be in place: first by date and then by index.
	- This is needed especially for `operation_columns*()` functions.

## Ledger structure
- Should all columns in the ledger have no spaces or are these allowed, used to keep consistency.
- Maybe the description column is redundant and the name of the operation is enough. Or maybe be more explicit in the description, instead of 'Buy {instrument}' it could be 'Buy X {instrument} in exchange for Y {price_in}' where X and Y are amounts.
- Maybe add a description to metadata? (I.e. what does a given instrument represent. E.g.: FSTFX is called "Fidelity Limited Term Municipal Income Fund" but from that name I don't necessarily know what that instrument is so a small description could be added to the metadata {a part from name and type} to allow easier instrument tracking).
- Maybe the operation column in the ledger should be called differently, perhaps action. The reason being that operations might be the things that are done via the API and they might have two events happening such as sell (selling and univesting).

### Undecided
- origin/destination/allocation
	- Possibly, origin and destination should always be filled. For instance in a sell operation, origin for the sell part would be the instrument itself.
	- Maybe there could be a column to state if an operations is just changing allocation, extracting or inserting into portfolio. Or maybe there should be an equivalence table to figure this out without having it on the ledger.
- **Tolerance**: Should the tolerance_decimals be set on a ledger level or operation level.
	- Maybe bring back a column 'Q_price_commission_tax_verification' with a different name and a different purpose. The column was removed to show where there were inconsistencies between a calculated and stated total, now that's not possible since an error is raised if such an inconsistency occurs if the `tolerance_decimals=4,` is surpassed. But since we are now using a tolerance setting to allow small inconsistencies, maybe a new column would be useful to showcase that small inconsistency.
	- Maybe `tolerance_decimals` should be called differently to allow inconsistencies of more than decimals, let's say 100 for instance.

## Operations review
- Deposit/withdraw
    - If there's a cost it should be stated as another operation
    - No commission/tax, add another op for that
    - Only an amount should be deposited or withdrew
- Add operations:
    - 'short' operation to allow to sell something I don't have.
    - 'borrow' to allow using money I don't have.
- There should be cost operations, probably with a 'pay_' prefix:
	- pay_deposit
	- pay_withdraw
	- pay_tax
	- pay_transfer
	- account_cost

## Logic review
- Validation
	- All operations should use positive numbers and raise exceptions if it's not the case. Maybe do it in add_rows.
	- Add validation for _add_rows
- Verify before operations if it's possible to do it. For instance
    - Sell only what I have and nothing more.
    - Invest only what I have and nothing more.
- In _cols_operation_balance_by_instrument_for_group(), for withdraw and sell there should be a final review. If withdraw/sell is more than what I have deposit should have a negative number to show an over withdraw or sell.
- Create a function that returns the last price in the ledger. This is meant to do calculation in case a current price cannot be found online or manually.
- When computing a balance, if a given instrument was bought using different instruments (e.g. I bought USD using EUR and also CHF), that means that I invested EUR and CHF into USD.
	- **Problem**:
		1. How to compute the current unrealized return.
		2. How to compute the return of a sold instrument into another currency. (e.g. I bought USD in CHF but sold them in EUR {or sold in another instrument}).
	- **The objective**: to know at a given date how much those USD have grown for me in terms of EUR and/or Fr (like a stock, the amount I posses doesn't change in time {unless I buy/sell/dividend stock}, what changes is its value).
	- **Possible solutions**:
		- Compute the value of all the used instruments in one of them at the time of buying. In our example, I used EUR and CHF to buy USD, so I could compute the amount of EUR payed in CHF to keep all in CHF (or vice versa) at the time I used those EUR to buy USD. So at the buying time, I bought USD with EUR, how many CHF do those EUR represent. This could make sense in the case I wanted to sell the USD in CHF.
		- Calculate all instruments into a given instrument (e.g. calculate all instruments in USD) at the initial point in time. Then calculare all instruments into a give instrument at the final time. Use those two values to compute the return in that given time.
- For `opid`:
	- it should be possible to insert an operation at a given point (the point indicated by an `opid`) which would replace that `opid` and increment it.
	- it should be possible to remove an `opid` and decrease all next operations to keep an `opid` consistent.
	- **or** maybe the operation id should be kept always to avoid confusions and consistency.

## Output formatting

> How functions behave with respect to the user interaction and their returns.

- In `SimplePortfolioLedger.ledger()` change the parameters `thousands_fmt_sep` and `thousands_fmt_decimals` so that it behaves like `some_pd_tools.pd_format.number_separators()`.
- In `SimplePortfolioLedger.ledger()` and `SimplePortfolioLedger.to_excel()` add sorting options, column to sort by and wether it's asc or desc.
- Review _ledger_columns_attrs.

## Testing
- Create unit tests for operations.
- Create automated unit tests:
    - Test if operations work as expected.
    - Test cols_operation* methods.

## To think about
- Maybe `simply_dtypes()` should happen only on setting the ledger and on operation (buy,sell,...) and on `_cols_operation_*` as those are computed on the fly. This would mean that setting `self._ledger_df` would need a setter, although this means that modifying the ledger probably would pass above this setter and it could be changed, if this is true, it's best to compute on reading the `simplify_dtypes()`. Or maybe this operation should happen both on setting and on reading, to allow better performance based on well selected dtypes and on reading to sanitize.

