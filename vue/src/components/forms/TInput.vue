<template>
  <div class="t-input">
    <label class="t-input__label" :style="{color: labelColor}">{{ label }}</label>
    <input v-model="input" class="t-input__input" :type="type" :name="parse" :value="value" @blur="$emit('blur', $event.target.value)" />
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
    labelColor: {
      type: String,
      default: 'A7BED3'
    },
  },
  data: () => ({
    input: ''
  })
};
</script>

<style lang="scss" scoped>
.t-input{
  &__label{
    color: #A7BED3;
    display: block;
    margin: 0 0 10px 0;
    line-height: 1.25;
    font-size: .75rem;
    user-select: none;
  }
  &__input{
    color: #fff;
    border-color: #6C7883;
    background: #242F3D;
    width: 100%;
  }
}
</style>
