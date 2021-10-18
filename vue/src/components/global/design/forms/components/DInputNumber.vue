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
import fields from '../../mixins/fields';
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
      this.$emit('change', { name: this.name, value });
      this.$emit('input', value);
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
@import '../scss/fields.scss';
.d-input {

}
</style>
