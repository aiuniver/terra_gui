<template>
  <div :class="['t-select', { 't-select--active': show }, { 't-select--small': small, 't-select--disabled': isDisabled }]"
  :style="`width: ${width}`"
  v-click-outside="outside">
    <div class="t-select__btn" @click="click">
      <i :class="['t-select__icon t-icon icon-file-arrow', { 't-select__icon--rotate': show }]"></i>
    </div>
    <input
      class="t-select__input"
      v-model="search"
      readonly
      :name="name"
      :disabled="isDisabled"
      :placeholder="placeholder || ''"
      :autocomplete="'off'"
      @click="click"
      @blur="select(false)"
      @focus="$emit('focus', $event)"
    />
    <label :for="name">{{ inputLabel }}</label>
    <div class="t-select__content" v-show="show">
      <div class="t-select__content--item" v-for="(item, i) in filterList" :key="i" @mousedown="select(item)" :title="item.label">
        {{ item.label }}
      </div>
      <div class="t-select__content--empty" v-if="!filterList.length">Нет данных</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-select',
  props: {
    type: String,
    placeholder: String,
    value: {
      type: [String, Number],
      default: '',
    },
    name: String,
    parse: String,
    inputLabel: String,
    list: {
      type: Array,
      default: () => [],
    },
    disabled: [Boolean, Array],
    small: Boolean,
    error: String,
    update: Boolean, //wtf
    width: String
  },
  data() {
    return {
      selected: {},
      show: false,
      input: ''
    };
  },
  created() {
    // console.log(this.list)
    // console.log(this.value)
    const list = this.list ?? [];
    this.selected = list.find(item => item.value === this.value) || {};
    if (this.update) {
      this.send(this.value); //wtf
    }
  },
  computed: {
    isDisabled() {
      if (Array.isArray(this.disabled)) {
        return !!this.disabled.includes(this.name);
      } else {
        return this.disabled;
      }
    },
    filterList() {
      return this.list ?? [];
    },
    search: {
      set(value) {
        this.input = value;
      },
      get() {
        const list = this.list ?? [];
        const label = list.find(el => el.value === this.selected.value) || null;
        return label ? label.label : '';
      },
    },
  },
  methods: {
    send(value) {
      this.$emit('input', value);
      this.$emit('change', { name: this.name, value });
      this.$emit('parse', { name: this.name, parse: this.parse, value });
    },
    label() {
      this.show = !this.show;
    },
    outside() {
      this.show = false;
    },
    select(item) {
      if (item) {
        this.selected = item;
        this.send(item.value);
        this.input = item.value;
      } else {
        // console.log(this.selected)
        // console.log(this.selected)
        this.search = this.selected.label || this.value || '';
      }
      this.show = false;
    },
    click(e) {
      this.show = !this.show;
      this.$emit('click', e);
    },
  },
  watch: {
    search(value) {
      if (!value) {
        this.$emit('parse', { name: this.name, parse: this.parse, value });
      }
      // console.log(value)
    },
  },
};
</script>

<style lang="scss" scoped>
.t-select {
  position: relative;
  height: 42px;
  label {
    position: absolute;
    margin-left: 10px;
    margin-top: 6px;
    font-size: 12px;
    line-height: 12px;
    color: #a7bed3;
  }
  &__icon {
    position: absolute;
    top: 10px;
    right: 13px;
    width: 10px;
    transition-duration: 100ms;
    &--rotate {
      transform: rotate(180deg);
    }
  }
  &__btn {
    position: absolute;
    width: 36px;
    border-left: 1px solid #65B9F4;
    height: 100%;
    right: 0;
    cursor: pointer;
  }
  &__input {
    height: 42px;
    width: 100%;
    padding: 0 10px;
    font-size: 14px;
    font-weight: 400;
    text-overflow: ellipsis;
    overflow: hidden;
    padding: 0 36px 0 10px;
    border-radius: 4px;
    border: 1px solid #65B9F4;
    color: #fff;
    background: #242f3d;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
    cursor: pointer;
    &:focus {
      background: rgba(101, 185, 244, 0.15);
    }
    &:disabled {
      cursor: default;
      opacity: 0.35;
      border: 1px solid #242F3D;
    }
  }
  &__content {
    position: absolute;
    top: 41px;
    min-width: 100%;
    border: 1px solid #65B9F4;
    box-shadow: 0px -8px 34px 0px rgba(0, 0, 0, 0.05);
    border-radius: 0 0 4px 4px;
    z-index: 3;
    color: #a7bed3;
    background-color: #242f3d;
    max-height: 200px;
    overflow: auto;
    &--item {
      color: inherit;
      font-size: 14px;
      line-height: 24px;
      text-align: left;
      cursor: pointer;
      text-overflow: ellipsis;
      white-space: nowrap;
      overflow: hidden;
      padding: 0 10px;
      &:hover {
        color: #65B9F4;
        background-color: #1E2734;
      }
    }
    &--empty {
      color: inherit;
      font-size: 14px;
      line-height: 24px;
      padding: 0 10px;
      cursor: default;
    }
  }
  &--active &__input {
    border-radius: 4px 4px 0 0;
  }
  &--small {
    height: 30px;
    .t-select__btn {
      width: 24px;
    }
  }
  &--small &__input {
    height: 30px;
    font-size: 12px;
    padding: 0 24px 0 5px;
    line-height: 24px;
  }
  &--small &__icon {
    top: 3px;
    right: 7px;
  }
  &--small &__content {
    top: 29px;
    width: 100%;
    &--item {
      padding: 0 5px;
      font-size: 12px;
      line-height: 22px;
    }
    &--empty {
      font-size: 12px;
      padding: 0 5px;
      line-height: 22px;
    }
  }
  &--disabled &__btn {
    border-left-color: #242F3D;
  }
}
</style>
