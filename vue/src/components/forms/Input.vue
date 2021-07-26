<template>
  <div class="field-form field-inline field-reverse">
    <slot>
      <label :for="name">{{ label }}</label>
    </slot>
    <input :id="name" :type="type" :name="parse" :value="value" />
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
    parse: String,
    name: String,
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
