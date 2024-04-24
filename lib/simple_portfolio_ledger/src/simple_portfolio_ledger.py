import pandas as pd
import datetime


class SimplePortfolioLedger:
    """_summary_

    Returns:
        _type_: _description_
    """

    # # Basic
    # 'date_execution', 'date_order',
    # 'operation',
    # 'instrument_type', 'name', 'instrument', 'origin', 'destination',
    # 'price_in', 'price', 'price_w_expenses', 'size', 'commission', 'tax', 'stated_total',
    # 'commission_notes', 'tax_notes',
    # # Additional
    # 'description', 'notes',
    # # Multiple accounts
    # 'account',
    # # Debugging
    # 'Q_price_commission_tax_verification',
    #
    # Should be the same columns as above but in a different order, more logical order
    _ledger_columns = [
        'date_execution',
        'operation',
        'instrument',
        'origin',
        'destination',
        'price_in',
        'price',
        'price_w_expenses',
        'size',
        'commission',
        'tax',
        'stated_total',
        'date_order',
        'name',
        'instrument_type',
        'description',
        'notes',
        'commission_notes',
        'tax_notes',
        'account',
        'Q_price_commission_tax_verification',
    ]

    # TODO: review attrs
    # Column description in a dict
    _ledger_columns_attrs = {
        'column notes': {
            'date_execution': 'The date the transaction was actually done.',
            'date_order': 'The date the transaction was created.',
            #
            # Ideas for operation:
            #   - reinvest (uninvest and invest in something else)
            #   - invest: change something into something else
            #   - uninvest:
            #
            'operation': 'Can be: buy, sell, deposit (for account), withdraw (for account)',
            'instrument_type': 'For example: cash, stock, mutual fund, etf, etc..',
            'name': 'Instrument name.',
            'instrument': 'Instrument code (USD, CLP, COP, AAPL, AMZN).',
            'origin': 'Where did the resource come from.',
            'destination': 'Where did the resource go.',
            'price_in': 'Instrument used for the operation.',
            'price': 'Price',
            'size': 'The amount in the operation.',
            'commission': '',
            'tax': '',
            'commission_notes' 'tax_notes': '',
            # Used to store additional information
            'description': 'What happened.',
            'notes': 'Additional notes for the transaction.',
            # These columns would allow me to have just one file for all my investments
            #   but might make the process more cumbersome
            'account': 'Account where the transaction was done.',
            # Debugging
            'Q_price_commission_tax_verification': '',
        }
    }

    def __init__(self) -> None:
        self._ledger_df = self._create_empty_ledger_df()

    @classmethod
    def get_ledger_columns(cls):
        return cls._ledger_columns

    @classmethod
    def whatis(cls, columns=None):
        """Small function to return information about different parts of the data.

        Args:
            columns (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if columns != None:
            if type(columns) == list:
                to_return = []
                for col in columns:
                    to_return.append({
                        col: cls._ledger_columns_attrs['column notes'].get(
                            col, 'NOT DEFINED')
                    })
                return to_return
            elif type(columns) == str:
                return cls._ledger_columns_attrs['column notes'].get(columns, 'NOT DEFINED')

    @classmethod
    def _create_empty_ledger_df(cls):

        # Create a portfolio_ledge with a row of dummies
        ledger = pd.DataFrame(
            [['dummy'] * len(cls._ledger_columns)],
            columns=cls._ledger_columns,
        )

        # About attrs: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.attrs.html
        # portfolio_ledger.attrs = portfolio_ledger_columns_attrs
        ledger.attrs = cls._ledger_columns_attrs

        # Empty portfolio_ledger
        ledger.drop(ledger.index, inplace=True)

        return ledger

    def ledger(
            self,
            cols_operation=False,
            cols_operation_cumsum=False,
            cols_operation_balance_by_instrument=False,
            thousands_fmt_sep=False,
            thousands_fmt_decimals=1
    ):
        if len(self._ledger_df) == 0:
            print('WARNING: Ledger is empty, showing only basic structure.')
            return self._ledger_df
        dfs_toconcat = [self._ledger_df]
        if cols_operation is True:
            dfs_toconcat.append(self.cols_operation())
        if cols_operation_cumsum is True:
            dfs_toconcat.append(self.cols_operation_cumsum())
        if cols_operation_balance_by_instrument is True:
            dfs_toconcat.append(self.cols_operation_balance_by_instrument())

        to_return = pd.concat(dfs_toconcat, join='inner', axis=1)

        if thousands_fmt_sep is True:
            return (to_return
                    .map(lambda x: f'{x:,.{thousands_fmt_decimals}f}' if isinstance(x, float) else x)
                    .map(lambda x: f'{x:,d}' if isinstance(x, int) else x)
                    )
        else:
            return to_return

    def cols_operation(self, instrument=False):
        """Returns a dataframe with 1 column per operation.

        Args:
            instrument (bool, optional): Whether or not to show the instrument. Defaults to False.

        Returns:
            pd.DataFrame: Returns a dataframe with 1 column per operation.
        """

        if len(self._ledger_df) == 0:
            return None

        # Create groups by instrument/operation
        groups = self._ledger_df.groupby(['instrument', 'operation'])

        # For each group, return the size (for example if an op is buy, the column buy would be filled, not all others)
        groups_opsize = groups.apply(lambda x: x['size'], include_groups=False)

        to_return = (
            groups_opsize
            # Move operation from the index and create one column for each op
            .unstack('operation')
            # Move index instrument created by the last groupby to a column
            .reset_index('instrument')
            .fillna(0)
            .sort_index()
        )

        if instrument is True:
            return to_return
        else:
            return to_return.drop(columns='instrument')

    # TODO: This needs to also group by account since the same instrument in different accounts has different cumsums
    def cols_operation_cumsum(self, show_inst_acc=False):
        """
        Add one column per operation but do a cumsum per instrument/operation.
        """

        if len(self._ledger_df) == 0:
            return None

        # Create groups by instrument/operation
        groups = self._ledger_df.groupby(
            ['operation', 'instrument', 'account'])

        # For each group do a cumsum for the size
        groups_cumsum = groups.apply(
            lambda x: x['size'].cumsum(), include_groups=False)

        # Used to remane the colums
        ops_names = groups_cumsum.unstack('operation').columns
        ops_cumsum_names = [f'cumsum {c}' for c in ops_names]
        rename_dict = dict(zip(ops_names, ops_cumsum_names))

        # Actual processing
        to_return = (
            groups_cumsum
            # Move operation from the index and create one column for each op
            .unstack('operation')
            # Rename created columns
            .rename(columns=rename_dict)
            # Move index 'instrument' and 'account' to a column
            .reset_index(['instrument', 'account'])
            # Regroup by instrument to do the ffill
            .groupby(['instrument', 'account'])
            # For each instrument, ffill with the cumsum, fill with 0 the beginning
            .apply(lambda x: x.ffill().fillna(0), include_groups=False)
            # Move indexes 'instrument' and 'account' created by the last groupby to a column
            .reset_index(['instrument', 'account'])
            # Create new colums
            .assign(
                # sum of all cumsum columns
                held=lambda x: x[ops_cumsum_names].sum(axis=1),
                # Invested total in positive value
                invested=lambda x: x[['cumsum invest',
                                      'cumsum uninvest']].sum(axis=1).abs(),
            )
            .rename(columns={'held': 'cumsum held', 'invested': 'cumsum invested'})
            .sort_index()
        )

        if show_inst_acc is False:
            to_return.drop(columns=['instrument', 'account'], inplace=True)

        return to_return

    @staticmethod
    def _cols_operation_balance_by_instrument_for_group(group_df, new_columns):
        """
        WARNING: not to be called by itself. It needs a grouping per instrument.
        """

        # Copy df so that no unexpected changes occur due to pointing logic
        df = group_df.copy()

        # Add new columns
        df[new_columns] = 0.0

        prev_operation_columns = new_columns.copy()
        # Remove from columns values that shouldn't be passed to the next line doing
        #   `prev_operation_balance = df[prev_operation_columns].iloc[0]`
        for coltoremove in ('sell profit loss', 'balance sell profit loss'):
            idxtoremove = prev_operation_columns.index(coltoremove)
            del prev_operation_columns[idxtoremove]

        prev_operation_balance = df[prev_operation_columns].iloc[0]

        for row in df.iterrows():
            idx, info = row

            # Copy previous operation balance (in next steps the current operation will be changed)
            #   This is necessary as some operations don't touch every new column
            #   and this assures the untouched operations keep track of history
            df.loc[idx, prev_operation_columns] = prev_operation_balance

            # TODO DELETE
            # Some operations below like:
            #     df.loc[idx, 'balance buy'] = df.loc[idx, 'balance buy'] + df.loc[idx, 'size']
            # Use the previous values set with the above code:
            #     df.loc[idx, prev_operation_columns] = prev_operation_balance

            # invest and uninvest should not add to any column as they are using the available instruments from a pool
            #   every dollar is equal. And for profit it's only important to know how much of a instrument was sold with a profit
            # elif info['operation'] == 'invest':
            #     df.loc[idx, 'invest'] = df.loc[idx, 'invest'] + df.loc[idx, 'size']
            # elif info['operation'] == 'uninvest':
            #     # For uninvest it should reduce the invest column
            #     df.loc[idx, 'invest'] = df.loc[idx, 'invest'] + df.loc[idx, 'size']

            if info['operation'] == 'deposit':
                df.loc[idx, 'balance deposit'] = (
                    prev_operation_balance['balance deposit'] +
                    df.loc[idx, 'size']
                )

            elif info['operation'] == 'buy':
                df.loc[idx, 'balance buy'] = (
                    prev_operation_balance['balance buy'] + df.loc[idx, 'size']
                )
                df.loc[idx, 'balance buy total payed'] = (
                    prev_operation_balance['balance buy total payed']
                    + df.loc[idx, 'stated_total']
                )
                df.loc[idx, 'avg buy total price'] = (
                    df.loc[idx, 'balance buy total payed']
                    / df.loc[idx, 'balance buy']
                )

            elif info['operation'] == 'dividend':
                df.loc[idx, 'balance dividend'] = (
                    prev_operation_balance['balance dividend'] +
                    df.loc[idx, 'size']
                )

            elif info['operation'] == 'stock dividend':
                df.loc[idx, 'balance stock dividend'] = (
                    prev_operation_balance['balance stock dividend'] +
                    df.loc[idx, 'size']
                )

            elif info['operation'] == 'withdraw':
                # withdraw takes money if available from this in this order: [deposit, stock dividend, dividend, buy]
                # withdraw is negative, use absolute
                withdrew = abs(df.loc[idx, 'size'])

                if withdrew > 0 and prev_operation_balance['balance deposit'] > 0:
                    if withdrew > prev_operation_balance['balance deposit']:
                        withdrew = withdrew - prev_operation_balance['balance deposit']
                        df.loc[idx, 'balance deposit'] = 0
                    else:
                        df.loc[idx, 'balance deposit'] = prev_operation_balance['balance deposit'] - withdrew
                        withdrew = 0
                if withdrew > 0 and prev_operation_balance['balance stock dividend'] > 0:
                    if withdrew > prev_operation_balance['balance stock dividend']:
                        withdrew = withdrew - prev_operation_balance['balance stock dividend']
                        df.loc[idx, 'balance stock dividend'] = 0
                    else:
                        df.loc[idx, 'balance stock dividend'] = prev_operation_balance['balance stock dividend'] - withdrew
                        withdrew = 0
                if withdrew > 0 and prev_operation_balance['balance dividend'] > 0:
                    if withdrew > prev_operation_balance['balance dividend']:
                        withdrew = withdrew - prev_operation_balance['balance dividend']
                        df.loc[idx, 'balance dividend'] = 0
                    else:
                        df.loc[idx, 'balance dividend'] = prev_operation_balance['balance dividend'] - withdrew
                        withdrew = 0
                if withdrew > 0:
                    df.loc[idx, 'balance buy'] = prev_operation_balance['balance buy'] - withdrew
                    df.loc[idx, 'balance buy total payed'] = (
                        prev_operation_balance['balance buy total payed'] -
                        withdrew * prev_operation_balance['avg buy total price']
                    )

                    # if df.loc[idx, 'balance buy'] > 0:
                    #     df.loc[idx, 'avg buy total price'] = (
                    #         df.loc[idx, 'balance buy total payed']
                    #         / df.loc[idx, 'balance buy']
                    #     )
                    # else:
                    #     df.loc[idx, 'avg buy total price'] = pd.NA

                # TODO WARNING:
                # There should be a final review, if withdraw is more than what I have, deposit should have a negative number to show an over withdraw or sell

            elif info['operation'] == 'sell':
                # sell takes money if available from this in this order: [deposit, stock dividend, dividend, buy]
                # sell is negative, use absolute
                sold = abs(df.loc[idx, 'size'])

                if sold > 0 and prev_operation_balance['balance deposit'] > 0:
                    if sold > prev_operation_balance['balance deposit']:
                        sold = sold - prev_operation_balance['balance deposit']
                        df.loc[idx, 'balance deposit'] = 0
                    else:
                        df.loc[idx, 'balance deposit'] = prev_operation_balance['balance deposit'] - sold
                        sold = 0
                if sold > 0 and prev_operation_balance['balance stock dividend'] > 0:
                    if sold > prev_operation_balance['balance stock dividend']:
                        sold = sold - prev_operation_balance['balance stock dividend']
                        df.loc[idx, 'balance stock dividend'] = 0
                    else:
                        df.loc[idx, 'balance stock dividend'] = prev_operation_balance['balance stock dividend'] - sold
                        sold = 0
                if sold > 0 and prev_operation_balance['balance dividend'] > 0:
                    if sold > prev_operation_balance['balance dividend']:
                        sold = sold - prev_operation_balance['balance dividend']
                        df.loc[idx, 'balance dividend'] = 0
                    else:
                        df.loc[idx, 'balance dividend'] = prev_operation_balance['balance dividend'] - sold
                        sold = 0
                if sold > 0:
                    df.loc[idx, 'balance buy'] = prev_operation_balance['balance buy'] - sold
                    df.loc[idx, 'balance buy total payed'] = (
                        prev_operation_balance['balance buy total payed'] -
                        sold * prev_operation_balance['avg buy total price']
                    )

                    # # Compute profit or loss
                    # sold_total_price = df.loc[idx, 'stated_total'] / df.loc[idx, 'size'] # Including commission and tax
                    profit_loss = (
                        sold * df.loc[idx, 'price_w_expenses']
                        - sold * prev_operation_balance['avg buy total price']
                    )
                    df.loc[idx, 'sell profit loss'] = profit_loss

            # For all operations, calculate 'avg buy total price'
            #   It might need to be pd.NA
            if df.loc[idx, 'balance buy'] == 0:
                df.loc[idx, 'avg buy total price'] = pd.NA
            else:
                df.loc[idx, 'avg buy total price'] = (
                    df.loc[idx, 'balance buy total payed']
                    / df.loc[idx, 'balance buy']
                )
            # df.loc[idx, 'cumsum sell profit loss'] = df.loc[idx, 'cumsum sell profit loss'] + profit_loss

            # TODO: WARNING:
            # There should be a final review, if withdraw is more than what I have, deposit should have a negative number to show an over withdraw or sell

            prev_operation_balance = df.loc[idx, prev_operation_columns]

        # Compute 'balance sell profit loss'
        df['balance sell profit loss'] = df['sell profit loss'].cumsum()

        return df

    # TODO: This needs to also group by account since the same instrument in different accounts has different balances
    def cols_operation_balance_by_instrument(self, show_inst_acc=False):

        if len(self._ledger_df) == 0:
            return None

        # Name not definitely defined: 'balance buy total payed' should represent the amount used to buy an instrument, for instance how much clp where used to buy USD
        new_columns = [
            'balance deposit',
            'balance dividend',
            'balance stock dividend',
            'balance buy',
            'balance buy total payed',
            'avg buy total price',
            # 'invest balance', # That was already calculated before, same as cumsum invest
            # 'uninvest balance', # uninvest balance doesn't make sense because this removes, we want invest balance, see previous note
            # 'withdraw', # withdraw balance doesn't make sense because it means removing something
            # 'sell', # sell balance doesn't make sense because it means removing something
            'sell profit loss',
            'balance sell profit loss'
        ]

        groups = (self
                  ._ledger_df
                  [['operation', 'instrument', 'price_w_expenses',
                      'size', 'stated_total', 'account']]
                  .groupby(['instrument', 'account']))

        to_return = (
            groups.apply(
                self._cols_operation_balance_by_instrument_for_group,
                include_groups=False,
                new_columns=new_columns,
            )
            # Move indexes 'instrument' and 'account' created by the last groupby to a column
            .reset_index(['instrument', 'account'])
            .sort_index()
        )

        if show_inst_acc is True:
            return to_return[['instrument', 'account', *new_columns]]
        else:
            return to_return[[*new_columns]]

    def rm_empty_rows(self):
        # Remove empty rows (where everything is filled with na)
        self._ledger_df.drop(
            self._ledger_df[self._ledger_df.isna().all(axis=1)].index,
            inplace=True
        )

    # *****************
    # Ledger Operations
    # *****************

    def deposit(
            self,
            date_execution,
            # operation,
            instrument,
            # origin,
            # destination,
            # price_in,
            # price,
            # price_w_expenses,
            size,
            commission,
            tax,
            stated_total,
            name,
            instrument_type,
            account,
            # Q_price_commission_tax_verification,

            # Optional arguments
            date_order=None,
            description='',
            notes='',
            commission_notes='',
            tax_notes='',
    ):

        # TODO:
        #   - If there was a cost for deposit/withdraw it should be stated as another operation
        #   - A deposit/withdraw shouldn't have a commission/tax because the cost is in another op, see above
        #   - A deposit/withdraw stated_total should be equal to size
        #   - Create some kind of enum to define
        #       - instrument_type
        #   - There should be cost operations, probably with a 'pay_' prefix:
        #       - pay_deposit
        #       - pay_withdraw
        #       - pay_tax
        #       - pay_transfer

        price = 1
        price_w_expenses = stated_total / size
        Q_price_commission_tax_verification = stated_total - \
            (size * price + commission + tax)
        new_deposit_row = {
            'date_execution': date_execution,
            'operation': 'deposit',
            'instrument': instrument,
            'origin': '',
            'destination': '',
            'price_in': instrument,
            'price': 1,
            'price_w_expenses': price_w_expenses,
            'size': size,
            'commission': commission,
            'tax': tax,
            'stated_total': stated_total,
            'date_order': date_order if date_order != None else date_execution,
            'name': name,
            'instrument_type': instrument_type,
            'description': description,
            'notes': notes,
            'commission_notes': commission_notes,
            'tax_notes': tax_notes,
            'account': account,
            'Q_price_commission_tax_verification': Q_price_commission_tax_verification,
        }
        new_df = pd.DataFrame([new_deposit_row])
        self._ledger_df = pd.concat([self._ledger_df, new_df])

    def invest(self):
        pass

    def buy(self):
        pass

    def sell(self):
        pass

    def uninvest(self):
        pass

    def withdraw(self):
        pass

    def dividend(self):
        pass

    def stock_dividend(self):
        pass
