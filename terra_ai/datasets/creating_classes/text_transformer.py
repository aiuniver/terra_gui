from terra_ai.datasets.creating_classes.base import BaseClass, PreprocessingTextClass
from terra_ai.datasets.data import DatasetInstructionsData, InstructionsData


class TextTransformerClass(PreprocessingTextClass, BaseClass):

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']

        parameters = None
        for inp_data in version_data.inputs:
            if inp_data.type == 'handler' and inp_data.parameters.type == 'Text':
                parameters = inp_data.parameters
        for out_data in version_data.outputs:
            if out_data.type == 'handler' and out_data.parameters.type == 'TextTransformer':
                out_data.parameters = parameters  # Тип TextTransformer меняется на Text!!!!!!!!!!!!

        return version_data

    def create_instructions(self, version_data, sources_temp_directory, version_paths_data):

        inp_data, inp_parameters = self.collect_data_to_pass(
            put_data=version_data.inputs,
            sources_temp_directory=sources_temp_directory,
            put_idx=0
        )

        inputs, inp_tags = self.create_put_instructions(
            dictio=inp_data,
            parameters=inp_parameters,
            version_sources_path=version_paths_data.sources
        )

        out_data, out_parameters = self.collect_data_to_pass(
            put_data=version_data.outputs,
            sources_temp_directory=sources_temp_directory,
            put_idx=len(inp_data) + 1  # ВНИМАНИЕ!
        )

        outputs, out_tags = self.create_put_instructions(
            dictio=out_data,
            parameters=out_parameters,
            version_sources_path=version_paths_data.sources
        )

        for key in outputs.keys():
            for col_name, data in outputs[key].items():
                output_instructions = outputs[key][col_name].instructions.copy()
                output_parameters = outputs[key][col_name].parameters.copy()
                raw_col_name = ' '.join(col_name.split('_')[1:])
                raw_id = len(inputs) + 1
                output_parameters['col_name'] = f"{raw_id}_{raw_col_name}_copy"
                output_parameters['cols_names'] = f"{raw_id}_{raw_col_name}_copy"
                output_parameters['put'] = raw_id
                for i in range(len(output_instructions)):
                    output_instructions[i] = f"[start] {output_instructions[i]} [end]"
                    outputs[key][col_name].instructions[i] = f"{outputs[key][col_name].instructions[i]} [end]"
                inputs.update({raw_id: {f"{raw_id}_{raw_col_name}_copy": InstructionsData(
                    instructions=output_instructions,
                    parameters=output_parameters)}}
                )

        instructions = DatasetInstructionsData(inputs=inputs, outputs=outputs)
        tags = {}
        tags.update(inp_tags)
        tags.update(out_tags)

        return instructions, tags
