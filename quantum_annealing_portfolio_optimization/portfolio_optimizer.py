from functools import cached_property
import os
import pandas as pd
from dimod import ConstrainedQuadraticModel, Binary, SampleSet
from dwave.system import LeapHybridCQMSampler
from canonical_transformer.morphisms import map_df_to_csv
from datetime import datetime
from .stock_info import StockInfo

class PortfolioOptimizer:

    TIME_LIMIT = 20.0
    BASE_DIR = 'data'
    FILE_FOLDER_RESULT = 'dataset-result'

    def __init__(self, stock_info: StockInfo, num_stocks_to_buy: int, budget: float, time_limit: float = TIME_LIMIT):
        self.stock_info = stock_info
        self.last_prices = self.stock_info.last_prices
        self.average_returns = self.stock_info.average_returns
        self.covariances = self.stock_info.covariances
        self.tickers = self.stock_info.tickers
        self.num_stocks_to_buy = self.set_num_stocks_to_buy(num_stocks_to_buy)
        self.budget = self.set_budget(budget)
        self.file_folder_result = os.path.join(self.BASE_DIR, self.FILE_FOLDER_RESULT)
        

    def set_num_stocks_to_buy(self, num_stocks_to_buy: int):
        if num_stocks_to_buy > len(self.tickers):
            raise ValueError(f'Number of stocks to buy must be less than or equal to the number of stocks available: {len(self.tickers)}')
        if num_stocks_to_buy < 1:
            raise ValueError(f'Number of stocks to buy must be greater than 0')
        if not isinstance(num_stocks_to_buy, int):
            raise TypeError(f'Number of stocks to buy must be an integer')
        return num_stocks_to_buy

    def set_budget(self, budget: float):
        if budget < 0:
            raise ValueError(f'Budget must be greater than 0')
        if not isinstance(budget, (float, int)):
            raise TypeError(f'Budget must be a float or an integer')
        if isinstance(budget, int):
            budget = float(budget)
        return budget

    def define_stocks(self) -> list[Binary]:
        """Define the stocks as binary variables."""
        return [Binary(f'stock_{ticker}') for ticker in self.tickers]

    @cached_property
    def stocks(self) -> list[Binary]:
        return self.define_stocks()

    def define_cqm(self) -> ConstrainedQuadraticModel:
        cqm = ConstrainedQuadraticModel()

        # Constraints
        cqm.add_constraint(sum(self.stocks) == self.num_stocks_to_buy, label='choose k stocks')
        cqm.add_constraint(sum(self.stocks * self.last_prices) <= self.budget, label='budget_limitation')

        # Objectives
        obj_1 = sum([-self.average_returns[i]*self.stocks[i] for i in range(len(self.stocks))])
        obj_2 = sum([self.covariances[i][j]*self.stocks[i]*self.stocks[j] for i in range(len(self.stocks)) for j in range(i+1, len(self.stocks))])
        cqm.set_objective(obj_1 + obj_2)

        return cqm

    @cached_property
    def cqm(self) -> ConstrainedQuadraticModel:
        return self.define_cqm()

    def sample_cqm(self, time_limit: float = TIME_LIMIT) -> SampleSet:
        """
        Sample the CQM using LeapHybridCQMSampler.
        
        Args:
            time_limit: Maximum time in seconds for the hybrid solver to run.
                       Default is 5.0 seconds. Increase for better solutions.
        
        Returns:
            SampleSet containing the solutions
        """
        sampler = LeapHybridCQMSampler()
        sampleset = sampler.sample_cqm(self.cqm, time_limit=time_limit)
        return sampleset

    @cached_property
    def sampleset(self) -> SampleSet:
        return self.sample_cqm()

    @cached_property
    def aggregate(self):
        return self.sampleset.aggregate()

    @cached_property
    def aggregated_record(self) -> list[tuple[list[int], float, int, list[bool], bool]]:
        return self.aggregate.record

    def get_df_from_record(self, record: list[tuple[list[int], float, int, list[bool], bool]], option_save_label: str = None) -> pd.DataFrame:
        df = pd.DataFrame(data=[record[0] for record in record], columns=self.tickers)
        df['energy'] = [record[1] for record in record]
        df['num_occurrences'] = [record[2] for record in record]
        df['constraints'] = [record[3] for record in record]
        df['feasible'] = [record[4] for record in record]

        if option_save_label:
            file_name = f'dataset-result-{option_save_label}-budget{self.budget}-n{self.num_stocks_to_buy}-save{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.csv'
            map_df_to_csv(df, file_folder=os.path.join(self.file_folder_result, file_name))

        return df

    @cached_property
    def df_record(self) -> pd.DataFrame:
        return self.get_df_from_record(self.sampleset.record, option_save_label='record')

    @cached_property
    def df_aggregated_record(self) -> pd.DataFrame:
        return self.get_df_from_record(self.aggregate.record, option_save_label='aggregate')

    def process_sampleset(self):
        """Read in sampleset returned from sample_cqm command and display solution."""

        # Find the first feasible solution
        first_run = True
        feasible = False
        best_sample = None
        best_energy = float('inf')
        for sample, energy, feas in self.sampleset.data(fields=['sample','energy','is_feasible']):
            if first_run:
                best_sample = sample
                best_energy = energy
            if feas:
                best_sample = sample
                best_energy = energy
                feasible = True
                break

        # Print the solution as which stocks to buy
        print("Solution:\n")
        if not feasible:
            print("No feasible solution found.\n")
        else:
            print("Best feasible solution found:")
            for stk in self.tickers:
                if best_sample[f'stock_{stk}'] == 1:
                    print(stk)
            print(f"Energy: {best_energy}")
        print("\n")
