<template>
  <div class="">
      <label>{{ label }}</label>
    <input :type="type" :name="parse" :value="value" />
  </div>
</template>

<script>
export default {
  name: 't-input',
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
    parse: {
      type: String,
    },
  },
};
</script>

<style lang="scss" scoped>
.field-form {
  input {
    max-width: 100px !important;
  }
}
</style>
