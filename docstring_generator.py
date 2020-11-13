    def read(cls, ticker):
        """[summary]

        Args:
            ticker ([type]): [description]

        Returns:
            [type]: [description]
        """
        original = pd.read_csv(os.path.join(data_path, file_prefix+ ticker + file_suffix))
        dates = original['Date'].map(lambda t: datetime.strptime(t, '%Y-%m-%d')).values
        prices = original['Open'].values
        original['dates'] = dates
        return cls(dates, prices, original)