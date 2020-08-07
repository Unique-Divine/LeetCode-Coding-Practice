    def getInfoGain(self, node, split_index):
        """[summary]

        Args:
            node ([type]): [description]
            split_index ([type]): [description]

        Returns:
            [type]: [description]
        """

        left_uncertainty = self.get_uncertainty(node.labels[:split_index]) 
        right_uncertainty = self.get_uncertainty(node.labels[split_index:])
        
        n = len(node.labels)
        w1 = left_uncertainty * (split_index+1) / n
        w2 = right_uncertainty * (n-(split_index+1))/n
        
        start_entropy = self.get_uncertainty(node.labels)
        conditional_entropy = w1 + w2 
        infogain = start_entropy - conditional_entropy
        return infogain

    def get_uncertainty(self, labels, metric="gini"):
        """[summary]

        Args:
            labels ([type]): [description]
            metric (str, optional): [description]. Defaults to "gini".

        Returns:
            [type]: [description]
        """        
        
        if labels.shape[0] == 0:
            return 1
        if metric =="gini":    
            uncertainty = gini(labels)
        if metric == "entropy":
            uncertainty = entropy(labels)
        
        return uncertainty