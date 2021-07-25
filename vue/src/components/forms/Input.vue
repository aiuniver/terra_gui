<template>
  <div class="field-form field-inline field-reverse">
    <slot>
      <label>{{ label }}</label>
    </slot>
    <input :type="type" :name="parse" :value="value" />
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
