<template>
  <div :class="['t-field', { 't-field--inline': inline }]">
    <div class="t-field__label flex align-center" @click="click">
      <template v-if="icon">
        <i class="ci-icon" :class="`ci-${icon}`"/>
      </template>
      <span>
        <slot name="label">{{ label }}</slot>
      </span>
    </div>
    <div class="t-field__input">
      <slot></slot>
    </div>
    <div v-if="error" class="t-field__hint">{{ error }}</div>
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
    icon: {
      type: String,
      default: '',
    },
    inline: Boolean,
  },
  data: () => ({
    error: '',
  }),
  methods: {
    click() {
      if (this.$children?.[0]?.label) this.$children[0].label();
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
  transition: all 0.2s ease-in-out;
  &__label {
    flex: 1 1 auto;
    color: #a7bed3;
    font-size: 0.75rem;
    padding: 0 0 10px 0;
    cursor: default;
    user-select: none;
    text-overflow: ellipsis;
    line-height: 1;
    & .ci-icon {
      font-size: 20px;
      margin-right: 5px;
    }

  }
  &--inline {
    flex-direction: row-reverse;
    justify-content: flex-end;
    align-items: center;
    .t-field__label {
      padding: 0 0 0 10px;
    }
  }
  &__hint {
    display: none;
    position: absolute;
    right: 0;
    top: calc(100% + 1px);
    max-width: 100%;
    word-wrap: break-word;
    background-color: #ca5035;
    // opacity: 0.9;
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.25);
    border-radius: 4px;
    font-size: 12px;
    line-height: 12px;
    padding: 5px 10px;
  }
  &:hover &__hint {
    display: block;
  }
}
</style>
