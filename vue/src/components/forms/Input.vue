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

