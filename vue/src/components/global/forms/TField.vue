<template>
  <div :class="['t-field', { 't-field--inline': inline }]">
    <div class="t-field__label" @click="click">
      <slot name="label">{{ label }}</slot>
    </div>
    <div class="t-field__input">
      <slot></slot>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-field',
  props: {
    label: {
      type: String,
      default: 'Label',
    },
    inline: Boolean,
  },
  methods: {
    click() {
      if (this.$children?.[0]?.label) {
        this.$children[0].label();
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-field {
  position: relative;
  display: flex;
  flex-direction: column;
  margin-bottom: 10px;
  &__label {
    flex: 0 0 auto;
    color: #a7bed3;
    line-height: 1.5;
    font-size: 0.75rem;
    padding: 0 0 10px 0;
    cursor: default;
    user-select: none;
  }
  &__input {
  }
}
.t-field {
  &--inline {
    flex-direction: row-reverse;
    justify-content: flex-end;
    align-items: center;
    .t-field__label {
      padding: 0 0 0 10px;
    }
    // .t-field__input {
    // flex: 0 0 109px;
    // }
  }
}
</style>
