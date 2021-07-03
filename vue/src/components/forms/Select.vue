<template>
  <div class="field-form field-inline field-reverse">
    <!-- Dropdown Input -->
    <label>{{ label }}</label>
    <select
      id="field_form-inputs[input_1][parameters][folder_name]"
      data-value-type="string"
      :name="name"
      style="display: none"
      :value="select"
    >
      <option value="" selected="selected"></option>
      <option value="Аппендицит">Аппендицит</option>
      <option value="Гастрит">Гастрит</option>
    </select>

    <at-select v-model="select" clearable size="small" :value="value" style="width: 100px">
      <at-option v-for="(item, i) of lists" :key="i" :value="item">{{ item }}</at-option>
    </at-select>
  </div>
</template>

<script>
export default {
  name: "Dropdown",
  props: {
    label: {
      type: String,
      default: "Label",
    },
    type: {
      type: String,
      default: "text",
    },
    value: {
      type: [String, Number],
    },
    name: {
      type: String,
    },
    parse: {
      type: String
    },
    lists: {
      type: Array,
      default: () => []
    }
  },
  data: () => ({
    select: ''
  })
}
</script>


<style lang="scss" scope>

</style>
