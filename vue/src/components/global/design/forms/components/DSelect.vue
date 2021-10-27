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
      <div :class="['d-input__btn--down', { 'd-input__btn--disabled': isDisabled }]">
        <i class="ci-icon ci-caret_down" />
      </div>
    </div>
    <div class="d-content">
      <template v-for="({ label, value }, i) of list">
        <div class="d-content__item" :key="label + i" @change="change({ label, value })">{{ label }}</div>
      </template>
    </div>
  </div>
</template>

<script>
import fields from '../../mixins/fields';
export default {
  name: 'd-select',
  mixins: [fields],
  props: {
    list: {
      type: Array,
      default: () => [],
    },
    type: {
      type: String,
      default: 'text',
    },
    placeholder: {
      type: String,
      default: 'Введите текст',
    },
    value: [String],
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
  &__btn {
    &--down {
      height: 100%;
      width: 35px;
      display: flex;
      justify-content: center;
      align-items: center;
      border-left: 1px solid;
      border-color: inherit;
      .ci-icon {
        font-size: 20px;
      }
    }
  }
}
.d-content {
  position: absolute;
  left: 0;
  top: calc(100% + 5px);
  background-color: #242f3d;
  min-width: 100%;
  border-radius: 4px;
  overflow: hidden;
  &__item {
    height: 36px;
    padding: 5px 10px;
    cursor: pointer;

    &:hover {
      background-color: #0e1621;
      color: #65b9f4;
    }
  }
}
</style>
