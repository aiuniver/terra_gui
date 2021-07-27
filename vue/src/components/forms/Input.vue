<template>
  <div class="field">
    <label class="field__label" :for="name">{{ label }}</label>
    <input class="field__input" :id="name" :type="type" :name="parse" :value="value" />
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
.field {
  &__label {
    color: #a7bed3;
    display: block;
    margin: 0 0 10px 0;
    line-height: 1.25;
    font-size: 0.75rem;
  }
  &__input {
    // max-width: 100px !important;
  }
}
</style>
