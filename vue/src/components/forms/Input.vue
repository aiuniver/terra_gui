<template>
  <div class="field-form field-inline field-reverse">
    <label>{{ label }}</label>
    <input
      :type="type"
      name="outputs[output_1][parameters][height]"
      :value="value"
      data-value-type="number"
    />
  </div>
</template>

<script>
export default {
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
    }
  },
};
</script>

