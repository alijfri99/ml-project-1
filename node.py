class node:
    def __init__(self, data, current_attrib, parent_attribs):
        self.data = data
        self.current_attrib = current_attrib
        self.parent_attribs = parent_attribs
        self.children = []
        self.entropy = None
        self.most_common_value = None
        self.most_common_result = None

    def determine_best_attribute(self, columns):
        for col in columns:
            if col in self.parent_attribs:
                continue
            else:
                for val in self.data[col].unique():
                    temp_child = node(self.data.loc[col == val], {col: val}, "To be determined later")
