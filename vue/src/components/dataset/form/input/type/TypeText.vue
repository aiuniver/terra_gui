<template>
  <v-row>
    <v-col cols="12" xl="6">
      <v-select
        :name="`inputs[${name}][parameters][folder_name]`"
        :items="folder_name_available"
        :value="folder_name"
        label="Folder name"
        outlined
        dense
        hide-details
      ></v-select>
    </v-col>
    <v-col cols="12" xl="6">
      <v-text-field
        :name="`inputs[${name}][parameters][delete_symbols]`"
        :type="delete_symbols_type"
        :value="delete_symbols"
        label="Delete sumbols"
        dense
        hide-details
        :rules="[rules['length_3']]"
        outlined
      ></v-text-field>
    </v-col>
    <v-col cols="12" xl="6">
      <v-text-field
        :name="`inputs[${name}][parameters][max_words_count]`"
        :type="max_words_count_type"
        :value="max_words_count"
        label="Max words count"
        dense
        hide-details
        :rules="[rules['required']]"
        outlined
      ></v-text-field>
    </v-col>
    <v-col cols="12" xl="6">
      <v-text-field
        :name="`inputs[${name}][parameters][step]`"
        :type="step_type"
        :value="step"
        label="Step"
        dense
        hide-details
        :rules="[rules['required']]"
        outlined
      ></v-text-field>
    </v-col>
    <v-col cols="12" xl="6">
      <v-text-field
        :name="`inputs[${name}][parameters][word_to_vec_size]`"
        :type="word_to_vec_size_type"
        :value="word_to_vec_size"
        label="Word to vec size"
        dense
        hide-details
        :rules="[rules['required']]"
        outlined
      ></v-text-field>
    </v-col>
    <v-col cols="12" xl="6">
      <v-text-field
        :name="`inputs[${name}][parameters][x_len]`"
        :type="x_len_type"
        :value="x_len"
        label="X len"
        dense
        hide-details
        :rules="[rules['required']]"
        outlined
      ></v-text-field>
    </v-col>
    <v-col cols="12" xl="6">
      <v-checkbox
        :name="`inputs[${name}][parameters][bag_of_words]`"
        label="Bag of words"
        :value="bag_of_words"
        true-value="true"
        false-value="false"
        dense
        hide-details
      ></v-checkbox>
    </v-col>
    <v-col cols="12" xl="6">
      <v-checkbox
        :name="`inputs[${name}][parameters][embedding]`"
        label="Embedding"
        :value="embedding"
        true-value="true"
        false-value="false"
        dense
        hide-details
      ></v-checkbox>
    </v-col>
    <v-col cols="12" xl="6">
      <v-checkbox
        :name="`inputs[${name}][parameters][word_to_vec]`"
        label="Word to vec"
        :value="word_to_vec"
        true-value="true"
        false-value="false"
        dense
        hide-details
      ></v-checkbox>
    </v-col>
    <v-col cols="12" xl="6">
      <v-checkbox
        :name="`inputs[${name}][parameters][pymorphy]`"
        label="Pymorphy"
        :value="pymorphy"
        :true-value="true"
        :false-value="false"
        dense
        hide-details
      ></v-checkbox>
    </v-col>
    <v-col cols="12" xl="6">
      <input
        type="checkbox"
        name="params[extra][use_bias]"
        checked="checked"
        data-value-type="boolean"
        data-unchecked-value="false"
      />
    </v-col>
  </v-row>
</template>

<script>
export default {
  props: {
    name: {
      type: String,
      default: "",
    },
    settings: {
      type: Object,
      default: () => {},
    },
  },
  data: () => ({
    folder_name: "",
    folder_name_available: [],

    delete_symbols: "",
    delete_symbols_type: "",
    max_words_count: "",
    max_words_count_type: "",
    step: "",
    step_type: "",
    word_to_vec_size: "",
    word_to_vec_size_type: "",
    x_len: "",
    x_len_type: "",

    bag_of_words: false,
    embedding: false,
    word_to_vec: false,

    pymorphy: false,

    rules: {
      length: (len) => (v) => (v || "").length >= len || `Length < ${len}`,
      required: (len) => len.length !== 0 || `Not be empty`,
      length_3: (len) => len.length > 2 || `Not be empty`,
    },
  }),
  mounted() {
    this.$nextTick(() => {
      const { text } = { ...this.settings };
      if (text) {
        for (const key in text) {
          if (text[key].default) {
            this[key] = text[key].default;
          }
          if (text[key].type && text[key].type !== "bool") {
            this[key + "_type"] = text[key].type === "int" ? "number" : "text";
          }
          if (text[key].available) {
            this[key + "_available"] = text[key].available;
          }
        }
      }
    });
  },
};
</script>
