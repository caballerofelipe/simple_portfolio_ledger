import pandas as pd
import datetime
import functools
import warnings
import inspect

"""
Notes:
- MARK is a functionality from Visual Studio Code to find things in the minimap.
"""

# TODO
_ = '''
- For all operations
    - Add type validation
    - Create some kind of enum to define
        - instrument_type
- Deposit/withdraw
    - If there's a cost should be stated as another operation
    - No commission/tax, add another op for that
    - Only an amount should be deposited or withdrew
IDEAS: 
- There should be cost operations, probably with a 'pay_' prefix:
    - pay_deposit
    - pay_withdraw
    - pay_tax
    - pay_transfer
    - account_cost
- Maybe add a new column for operation id and sub id?
    - For instance, a sell is an univest and a sell, so both ops should have an id
        e.g. 132 and maybe a sub id 1 and 2
        these could go on different columns or in one column (132-1 and 132-2)
- Maybe date_execution and date_order could be datetime instead of only date
'''


class SimplePortfolioLedger:
    """_summary_

    Returns:
        _type_: _description_
    """

    # # Basic
    # 'date_execution', 'date_order',
    # 'operation',
    # 'instrument_type', 'instrument_name', 'instrument', 'origin', 'destination',
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
    _ledger_columns = (
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
        'instrument_name',
        'instrument_type',
        'description',
        'notes',
        'commission_notes',
        'tax_notes',
        'account',
        'Q_price_commission_tax_verification',
    )

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
            'instrument_name': 'Instrument name.',
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

    _ops_names = set(('buy', 'deposit', 'dividend', 'invest',
                     'sell', 'stock dividend', 'uninvest', 'withdraw'))

    def __init__(self) -> None:
        self._ledger_df = self._create_empty_ledger_df()

    # *****************
    # Decorators
    # MARK: Decorators
    # *****************

    @staticmethod
    def _deco_check_ledger_for_cols(func):
        # https://stackoverflow.com/a/72563047/1071459
        deco_name = inspect.stack()[0].function

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):

            if len(self._ledger_df) == 0:
                warnings.warn(
                    f'WARNING: Ledger is empty, no columns computed, showing only basic column structure. (Warning issued by decorator [{deco_name}])', stacklevel=2)
                # return None

            extraneous_ops = self._get_extraneous_ops()
            if len(extraneous_ops) > 0:
                raise ValueError(
                    'One or more forbidden operations were inserted in the ledger.',
                    extraneous_ops)

            return func(self, *args, **kwargs)

        return wrapper

    # *****************
    # Classmethods
    # MARK: Classmethods
    # *****************

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

    def _get_extraneous_ops(self):
        # Compute extraneous_ops, which shows all unique operations from self._ledger_df
        #   minus all operations self._ops_names
        #   leaving operations that shouldn't be here
        used_ops = set(self._ledger_df['operation'].unique())
        extraneous_ops = used_ops - set(self._ops_names)
        return extraneous_ops

    # *****************
    # Computing operations
    # MARK: Computing Operations
    # *****************

    def ledger(
            self,
            cols_operation=False,
            cols_operation_cumsum=False,
            cols_operation_balance_by_instrument=False,
            thousands_fmt_sep=False,
            thousands_fmt_decimals=1
    ):
        if len(self._ledger_df) == 0:
            warnings.warn(
                'WARNING: Ledger is empty, showing only basic structure.')
        dfs_toconcat = [self._ledger_df]
        if cols_operation is True:
            dfs_toconcat.append(self.cols_operation())
        if cols_operation_cumsum is True:
            dfs_toconcat.append(self.cols_operation_cumsum())
        if cols_operation_balance_by_instrument is True:
            dfs_toconcat.append(self.cols_operation_balance_by_instrument())

        to_return = (pd
                     .concat(dfs_toconcat, join='inner', axis=1)
                     .convert_dtypes(
                         convert_string=False
                     )
                     .astype({
                         'date_execution': 'datetime64[s]',
                         'date_order': 'datetime64[s]',
                     })
                     )

        if thousands_fmt_sep is True:
            return (to_return
                    .map(lambda x: f'{x:,.{thousands_fmt_decimals}f}' if isinstance(x, float) else x)
                    .map(lambda x: f'{x:,d}' if isinstance(x, int) else x)
                    )
        else:
            return to_return

    @_deco_check_ledger_for_cols
    def cols_operation(self, show_instr_accnt=False):
        """Returns a dataframe with 1 column per operation.

        Args:
            show_instr_accnt (bool, optional): Whether or not to show the instrument and the account. Defaults to False.

        Returns:
            pd.DataFrame: Returns a dataframe with 1 column per operation.
        """

        # Create groups by instrument/operation
        groups = self._ledger_df.groupby(
            ['instrument', 'account', 'operation'])

        # For each group, return the size (for example if an op is buy, the column buy would be filled, not all others)
        groups_opsize = groups.apply(lambda x: x['size'], include_groups=False)

        to_return = (
            groups_opsize
            # Move operation from the index and create one column for each op
            .unstack('operation')
            # Add columns that weren't created in the previous operation
            #   using the operation list from the class self._ops_names
            .reindex(sorted(self._ops_names), axis=1, fill_value=0)
            # Convert to the best possible dtypes using dtypes supporting pd.NA
            .convert_dtypes(convert_string=False)
            # Fill na with 0
            .fillna(0)
            # Move index instrument created by the last groupby to a column
            .reset_index(['instrument', 'account'])
            .sort_index()
        )

        if show_instr_accnt is True:
            return to_return
        else:
            return to_return.drop(columns=['instrument', 'account'])

    @_deco_check_ledger_for_cols
    def cols_operation_cumsum(self, show_instr_accnt=False):
        """
        Add one column per operation but do a cumsum per instrument/operation.
        """

        # Create groups by instrument/operation
        groups = self._ledger_df.groupby(
            ['operation', 'instrument', 'account'])

        # For each group do a cumsum for the size
        groups_cumsum = groups.apply(
            lambda x: x['size'].cumsum(), include_groups=False)

        # Used to remane the colums
        ops_cumsum_names = [f'cumsum {c}' for c in self._ops_names]
        rename_dict = dict(zip(self._ops_names, ops_cumsum_names))

        # Actual processing
        to_return = (
            groups_cumsum
            # Move operation from the index and create one column for each op
            .unstack('operation')
            # Add columns that weren't created in the previous operation
            #   using the operation list from the class self._ops_names
            .reindex(sorted(self._ops_names), axis=1, fill_value=0)
            # Convert to the best possible dtypes using dtypes supporting pd.NA
            .convert_dtypes(convert_string=False)
            # Fill na with 0
            .fillna(0)
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

        if show_instr_accnt is False:
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
        # Remove from columns values that shouldn't be passed to the line:
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
            #    used in conjunction with
            #       - The line outside this loop:
            #            `prev_operation_balance = df[prev_operation_columns].iloc[0]`
            #       - The last line from this loop:
            #           `prev_operation_balance = df.loc[idx, prev_operation_columns]`
            df.loc[idx, prev_operation_columns] = prev_operation_balance

            # invest and uninvest should not be part of the balance
            #   the invested balance is computed in `cols_operation_cumsum`, this applies to:
            # `info['operation'] == 'invest'` and `info['operation'] == 'uninvest'`

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

            # TODO: WARNING:
            # There should be a final review, if withdraw is more than what I have, deposit should have a negative number to show an over withdraw or sell

            prev_operation_balance = df.loc[idx, prev_operation_columns]

        # Compute 'balance sell profit loss'
        df['balance sell profit loss'] = df['sell profit loss'].cumsum()

        return df

    @_deco_check_ledger_for_cols
    def cols_operation_balance_by_instrument(self, show_instr_accnt=False):

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
            # Add columns that weren't created in the previous operation
            #   using the columns from `new_columns`
            .reindex(new_columns, axis=1, fill_value=0)
            # Convert to the best possible dtypes using dtypes supporting pd.NA
            .convert_dtypes(convert_string=False)
            # Fill na with 0
            .fillna(0)
            # Move indexes 'instrument' and 'account' created by the last groupby to a column
            .reset_index(['instrument', 'account'])
            .sort_index()
        )

        if show_instr_accnt is True:
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
    # MARK: Ledger Operations
    # *****************

    def _add_row(self, value_dict):
        self._ledger_df = pd.concat([
            self._ledger_df,
            pd.DataFrame([value_dict])
        ]).reset_index(drop=True)

    def buy(self):
        pass

    def deposit(self,
                date_execution, instrument, amount, instrument_name, instrument_type, account,
                # Optional arguments
                date_order=None, description='', notes='',
                ):
        """Creates a new row with a deposit.

        Important: Deposit means a instrument entering an account. A commission or tax for a deposit is not possible, create a new operation for that purpose after doing the deposit.

        Args:
            date_execution (_type_): _description_
            instrument (_type_): _description_
            amount (_type_): _description_
            instrument_name (_type_): _description_
            instrument_type (_type_): _description_
            account (_type_): _description_
            date_order (_type_, optional): _description_. Defaults to None.
            description (str, optional): _description_. Defaults to ''.
            notes (str, optional): _description_. Defaults to ''.
        """
        self._add_row({
            'date_execution': date_execution,
            'operation': 'deposit',
            'instrument': instrument,
            'origin': '',
            'destination': instrument,
            'price_in': instrument,
            'price': 1,
            'price_w_expenses': 1,
            'size': amount,
            'commission': 0,
            'tax': 0,
            'stated_total': amount,
            'date_order': date_order if date_order != None else date_execution,
            'instrument_name': instrument_name,
            'instrument_type': instrument_type,
            'description': description,
            'notes': notes,
            'commission_notes': '',
            'tax_notes': '',
            'account': account,
            'Q_price_commission_tax_verification': 0,
        })

    def dividend(self):
        pass

    def invest(self):
        pass

    def sell(self):
        pass

    def stock_dividend(self):
        pass

    def uninvest(self):
        pass

    def withdraw(self,
                 date_execution, instrument, amount, instrument_name, instrument_type, account,
                 # Optional arguments
                 date_order=None, description='', notes='',
                 ):
        self._add_row({
            'date_execution': date_execution,
            'operation': 'withdraw',
            'instrument': instrument,
            'origin': instrument,
            'destination': '',
            'price_in': instrument,
            'price': 1,
            'price_w_expenses': 1,
            'size': -amount,
            'commission': 0,
            'tax': 0,
            'stated_total': -amount,
            'date_order': date_order if date_order != None else date_execution,
            'instrument_name': instrument_name,
            'instrument_type': instrument_type,
            'description': description,
            'notes': notes,
            'commission_notes': '',
            'tax_notes': '',
            'account': account,
            'Q_price_commission_tax_verification': 0,
        })
