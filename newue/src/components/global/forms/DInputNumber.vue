<template>
  <div
    class="d-input"
    :class="[{ 'd-input--error': error }, { 'd-input--small': small }, { 'd-input--disabled': disabled }]"
  >
    <div v-if="icon" class="d-input__icon">
      <i :class="`ci-icon ci-${icon}`" />
    </div>
    <input
      v-model.number="input"
      v-bind="$attrs"
      autocomplete="false"
      :class="['d-input__input']"
      :type="type"
      :name="name"
      :placeholder="placeholder"
      :disabled="isDisabled"
      @input="debounce"
      @focus="focus"
      @change="change"
    />
    <div class="d-input__btn">
      <div v-if="type !== 'number'" v-show="input && !isDisabled" class="d-input__btn--cleener">
        <i class="ci-icon ci-close_big" @click="clear" />
      </div>
      <div v-if="type === 'number' && !disabled" :class="['d-input__btn--number']">
        <i class="ci-icon ci-caret_up" @click="caretClick(1)" />
        <i class="ci-icon ci-caret_down" @click="caretClick(-1)" />
      </div>
    </div>
  </div>
</template>

<script>
import fields from '@/mixins/forms/fields';
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
    disabled: Boolean,
    parse: [String, Number]
  },
  data: () => ({
    input: 0,
    tID: 0
  }),
  computed: {},
  methods: {
    clear() {
      this.input = 0;
      this.send(0);
    },
    focus(e) {
      this.$emit('focus', e);
    },
    change({ target }) {
      this.send(+target.value);
    },
    send(value) {
      this.$emit('input', value);
      this.$emit('change', { name: this.name, value });
      this.$emit('parse', { name: this.name, parse: this.parse, value });
    },
    caretClick(val) {
      clearTimeout(this.tID)
      this.input += val
      this.tID = setTimeout(() => {
        this.send(this.input)
      }, 300)
    }
  },
  created() {
    this.input = this.value;
  },
  watch: {
    value(value) {
      this.input = +value;
    },
  },
};
</script>

<style lang="scss" scoped>
@import '@/assets/scss/components/fields.scss';
.d-input {
  overflow: hidden;
  input {
    appearance: textfield;
    &::-webkit-inner-spin-button, &::-webkit-outer-spin-button {
      appearance: none;
    }
  }
}
</style>
