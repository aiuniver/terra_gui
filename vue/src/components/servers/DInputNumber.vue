<template>
  <div
    class="d-input"
    :class="[{ 'd-input--error': error }, { 'd-input--small': small }, { 'd-input--disabled': isDisabled }]"
  >
    <div v-if="icon" class="d-input__icon">
      <i :class="`ci-icon ci-${icon}`" />
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
      <!-- <div v-show="input && !isDisabled" class="d-input__btn--cleener">
        <i class="ci-icon ci-close_big" @click="clear" />
      </div> -->
      <div v-if="type === 'number'" :class="['d-input__btn--number', { 'd-input__btn--disabled': isDisabled }]">
        <i class="ci-icon ci-caret_up" @click="send(input+1)" />
        <i class="ci-icon ci-caret_down" @click="send(input-1)" />
      </div>
    </div>
  </div>
</template>

<script>
import fields from '@/components/global/design/mixins/fields.js';

export default {
  name: 'd-input-number',
  mixins: [fields],
  props: {
    type: {
      type: String,
      default: 'number',
    },
    placeholder: {
      type: String,
      default: 'Введите число',
    },
    value: [Number],
  },
  data: () => ({
    input: '',
  }),
  computed: {},
  methods: {
    clear() {
      this.input = '';
      this.send('');
    },
    focus(e) {
      this.$emit('focus', e);
    },
    change({ target }) {
      this.send(target.value);
    },
    send(value) {
      if (value < 0) return this.input = 0
      this.$emit('change', { name: this.name, value });
      this.$emit('input', +value);
    },
  },
  created() {
    this.input = this.value;
  },
  watch: {
    value(value) {
      this.input = value;
    },
  },
};
</script>

<style lang="scss" scoped>
@import '@/components/global/design/forms/scss/fields.scss';
.d-input {
  background: #242f3d;
  border: 1px solid #6c7883;
  &__btn--number .ci-icon:hover {
    background: none;
  }
  &__input {
    padding: 8px 10px;
  }
  &:not(.d-input--error):focus-within {
    background-color: #242f3d;
    border-color: #fff;
    box-shadow: none;
  }
  &:not(.d-input--disabled):not(.d-input--error):hover {
    border-color: unset;
    background: #242f3d;
  }
  &--disabled {
    opacity: 0.5;
  }
  &--error {
    border-color: $color-border-error;
    color: $text-color-error;
  }
}
</style>
