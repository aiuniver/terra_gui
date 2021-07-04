<template>
  <div class="field-form field-inline field-reverse">
    <label>{{ label }}</label>
    <div class="checkout-switch">
      <input
        :checked="value ? 'checked' : ''"
        data-value-type="boolean"
        data-unchecked-value="false"
        :type="type"
        :value="value"
        :name="`${parse}[parameters][${name}]`"
      />
      <span class="switcher"></span>
    </div>
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
      type: [String, Number, Boolean],
    },
    name: {
      type: String,
    },
    parse: {
      type: String,
    },
  },
};
</script>

