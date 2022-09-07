import numpy as np
from typing import Dict, Callable

from transformations import (
    BlockSwap, ConfusionRemover, DeadCodeInserter, ForWhileTransformer, OperandSwap, SyntacticNoisingTransformation
)


class SemanticPreservingTransformation:
    def __init__(
            self,
            parser_path: str,
            language: str,
            transform_functions: Dict[Callable, int] = None,
    ):
        self.language = language
        if transform_functions is not None:
            self.transform_functions = transform_functions
        else:
            self.transform_functions = {
                BlockSwap: 1,
                ConfusionRemover: 1,
                DeadCodeInserter: 1,
                ForWhileTransformer: 1,
                OperandSwap: 1,
                SyntacticNoisingTransformation: 1,
                VarRenamer: 1
            }
        self.transformations = []
        if self.language == "nl":
            self.transformations.append(SyntacticNoisingTransformation(parser_path=parser_path, language="nl"))
        else:
            for t in self.transform_functions:
                for _ in range(self.transform_functions[t]):
                    self.transformations.append(t(parser_path=parser_path, language=language))

    def transform_code(
            self,
            code: str
    ):
        transformed_code, transformation_name = None, None
        indices = list(range(len(self.transformations)))
        # np.random.shuffle(indices)
        # success = False
        # while not success and len(indices) > 0:
        #     si = np.random.choice(indices)
        #     indices.remove(si)
        #     t = self.transformations[si]
        #     transformed_code, metadata = t.transform_code(code)
        #     print(transformed_code, metadata)
        #     success = metadata["success"]
        #     if success:
        #         transformation_name = type(t).__name__
        # if not success:
        #     print('failed')
        #     return code, None
        for si in indices:
            t = self.transformations[si]
            code, metadata = t.transform_code(code)

        return code, transformation_name
