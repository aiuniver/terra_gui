<template>
  <div :class="['d-input', { 'd-input--error': error }, { 'd-input--small': small }]">
    <input
      v-model="input"
      v-bind="$attrs"
      autocomplete="off"
      :class="['d-input__input']"
      :type="type"
      :name="name"
      :placeholder="placeholder"
      :disabled="isDisabled"
      @input="debounce"
      @focus="focus"
    />
    <div class="d-input__btn" v-show="input && !isDisabled">
      <i class="ci-icon ci-close_big" @click="clear" />
    </div>
  </div>
</template>

<script>
import { debounce } from '@/utils/core/utils';
export default {
  name: 'd-input',
  props: {
    type: {
      type: String,
      default: 'text',
    },
    placeholder: {
      type: String,
      default: 'Введите текст',
    },
    value: [String, Number],
    disabled: [Boolean, Array],
    name: String,
    small: Boolean,
    error: String,
  },
  data: () => ({
    input: '',
    debounce: null,
  }),
  computed: {
    isDisabled() {
      return Array.isArray(this.disabled) ? !!this.disabled.includes(this.name) : this.disabled;
    },
  },
  methods: {
    clear() {
      this.input = '';
      this.send('');
    },
    label() {
      this.$el.children[0].focus();
    },
    focus(e) {
      this.$emit('focus', e);
    },
    change({ target }) {
      this.send(target.value);
    },
    send(value) {
      this.$emit('change', { name: this.name, value });
      this.$emit('input', value);
    },
  },
  created() {
    this.input = this.value;
    this.debounce = debounce(this.change, 300);
    if (this.$parent?.$options?._componentTag === 't-field') this.$parent.error = this.error;
  },
  watch: {
    value(value) {
      this.input = value;
    },
    error(value) {
      console.log(this.$parent?.$options?._componentTag);
      if (this.$parent?.$options?._componentTag === 't-field') this.$parent.error = value;
    },
  },
};
</script>

<style lang="scss" scoped>
.d-input {
  position: relative;
  &__input {
    width: 100%;
    height: 40px;
    color: #fff;
    background: #242f3d50;
    padding: 8px 10px;
    font-size: 14px;
    line-height: 24px;
    font-weight: 400;
    border-radius: 4px;
    border: 1px solid #242f3d;
    transition: all 0.2s ease-in-out;
    &:not([disabled]):hover,
    &:not([disabled]):focus {
      border-color: #65b9f4;
      box-shadow: 0px 0px 4px rgba(101, 185, 244, 0.2);
    }
    &::placeholder {
      color: #6c7883;
    }
    &:disabled {
      opacity: 0.5;
    }
  }
  &--small {
    .d-input__input {
      width: 100%;
      height: 30px;
      padding: 3px 10px;
    }
  }
  &--error {
    .d-input__input,
    .d-input__input:hover,
    .ci-icon {
      border-color: #ca5035;
      color: #ca5035;
    }
  }
  &__btn {
    position: absolute;
    top: 0;
    right: 0;
    height: 100%;
    width: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    .ci-icon {
      cursor: pointer;
    }
  }
}
</style>
