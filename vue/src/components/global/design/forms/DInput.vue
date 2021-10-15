<template>
  <div
    class="d-input"
    :class="[{ 'd-input--error': error }, { 'd-input--small': small }, { 'd-input--disabled': isDisabled }]"
  >
    <div class="d-input__icon">
      <i class="ci-icon ci-search"/>
    </div>
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
    <div class="d-input__btn">
      <div v-show="input && !isDisabled" class="d-input__btn--cleener">
        <i class="ci-icon ci-close_big" @click="clear" />
      </div>
      <div v-if="type === 'number'" :class="['d-input__btn--number', { 'd-input__btn--disabled': isDisabled }]">
        <i class="ci-icon ci-caret_up" @click="+input++" />
        <i class="ci-icon ci-caret_down" @click="+input--" />
      </div>
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
  border-color: #242f3d;
  border: 1px solid #242f3d;
  background: #242f3d50;
  border-radius: 4px;
  display: flex;
  width: 100%;
  height: 40px;
  color: #fff;
  transition: all 0.2s ease-in-out;
  overflow: hidden;
  &:not(.d-input--error):focus-within,
  &:not(.d-input--disabled):not(.d-input--error):hover {
    border-color: #65b9f4;
    box-shadow: 0px 0px 4px rgba(101, 185, 244, 0.2);
  }
  &--disabled {
    opacity: 0.5;
  }
  &--error {
    border-color: #ca5035;
    color: #ca5035;
  }
  &__icon{
    height: 100%;
    display: flex;
    align-items: center;
    padding: 0 0 0 5px;
  }
  &__input {
    height: 100%;
    padding: 8px 10px;
    font-size: 14px;
    line-height: 24px;
    font-weight: 400;
    border: none;
    background: none;
    color: inherit;
    &::placeholder {
      color: #6c7883;
    }
    :disabled {
      opacity: 1;
    }
  }
  &--small {
    height: 30px;
    .d-input__input {
      padding: 3px 10px;
    }
  }
  &__btn {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    border-color: inherit;
    .ci-icon {
      cursor: pointer;
    }
    &--cleener {
      margin: 0 5px 0 0;
    }
    &--number {
      height: 100%;
      width: 35px;
      display: flex;
      flex-direction: column;
      justify-content: space-around;
      border-left: 1px solid;
      border-color: inherit;
      .ci-icon {
        height: 100%;
        text-align: center;
        &:hover {
          background-color: #222e3b;
        }
      }
    }
    &--disabled {
      .ci-icon {
        &:hover {
          background: none;
          cursor: default;
        }
      }
    }
  }
}
</style>
