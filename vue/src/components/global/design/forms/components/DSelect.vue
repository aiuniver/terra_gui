<template>
  <div
    class="d-input"
    :class="[{ 'd-input--error': error }, { 'd-input--small': small }, { 'd-input--disabled': isDisabled }]"
    v-click-outside="outside"
  >
    <div v-if="icon" class="d-input__icon">
      <i :class="`ci-icon ci-${icon}`" />
    </div>
    <input
      v-model="search"
      v-bind="$attrs"
      autocomplete="off"
      readonly
      :class="['d-input__input']"
      :type="type"
      :name="name"
      :placeholder="placeholder"
      :disabled="isDisabled"
      @click="click"
      @focus="focus"
    />
    <div class="d-input__btn">
      <div v-show="selected.value && !isDisabled" class="d-input__btn--cleener">
        <i class="ci-icon ci-close_big" @click="clear" />
      </div>
      <div :class="['d-input__btn--down', { 'd-input__btn--disabled': isDisabled }]" @click="click">
        <i class="ci-icon ci-caret_down" />
      </div>
    </div>
    <div v-if="show" class="d-content">
      <template v-for="({ label, value }, i) of filterList">
        <div class="d-content__item" :key="label + i" @click="select({ label, value })">{{ label }}</div>
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
      default: 'Выбрать пункт',
    },
    value: [String],
  },
  data: () => ({
    selected: {},
    input: '',
    show: false,
  }),
  computed: {
    filterList() {
      return this.list;
    },
    search: {
      set(value) {
        this.input = value;
      },
      get() {
        return this.list.find(item => item.value === this.selected?.value)?.label || '';
      },
    },
  },
  methods: {
    outside() {
      this.show = false;
    },
    clear() {
      this.selected = {};
      this.send('');
    },
    focus(e) {
      this.$emit('focus', e);
    },
    click(e) {
      if (!this.isDisabled) {
        if (!this.show) {
          console.dir(this.$el.children[0].focus());
        }
        this.show = !this.show;
        this.$emit('click', e);
      }
    },
    select(item) {
      console.log(item);
      this.selected = item;
      this.send(item.value);
      this.show = false;
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
  &__input {
    cursor: default;
  }
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
  z-index: 5;
  &__item {
    height: 36px;
    padding: 5px 10px;
    cursor: pointer;
    color: #a7bed3;
    &:hover {
      background-color: #0e1621;

      color: #65b9f4;
    }
  }
}
</style>
