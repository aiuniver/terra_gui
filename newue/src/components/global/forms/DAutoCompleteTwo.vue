<template>
  <div :class="['t-auto-complete', { 't-auto-complete--active': show }, { 't-auto-complete--small': small }]" v-outside="outside">
    <i :class="['t-auto-complete__icon t-icon icon-file-arrow', { 't-auto-complete__icon--rotate': show }]" @click="click"></i>
    <input
      class="t-auto-complete__input"
      v-model="input"
      :name="name"
      :disabled="isDisabled"
      :placeholder="placeholder || ''"
      :autocomplete="'off'"
      @click="click"
      @blur="select(false)"
      @focus="$emit('focus', $event), $event.target.select()"
    />
    <label :for="name">{{ inputLabel }}</label>
    <div class="t-auto-complete__content" v-show="show">
      <div class="t-auto-complete__content--item" v-for="(item, i) in list" :key="i" @mousedown="select(item)">
        {{ item.label }}
      </div>
      <div class="t-auto-complete__content--empty" v-if="!list.length">Нет данных</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'd-auto-complete-two',
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
    all: Boolean,
    error: String,
    update: Boolean, //wtf
  },
  data() {
    return {
      selected: {},
      show: false,
      input: '',
    };
  },
  created() {
    const list = this.list ?? [];
    this.selected = list.find(item => item.value === this.value) || {};
    this.input = this.selected?.label || '';
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
  },
  methods: {
    send(value) {
      console.log('value', value);
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
      }
      this.show = false;
    },
    click(e) {
      this.show = !this.show;
      this.$emit('click', e);
    },
  },
};
</script>

<style lang="scss">
.t-auto-complete {
  position: relative;
  height: 40px;
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
    right: 10px;
    width: 8px;
    cursor: pointer;
    transition-duration: 100ms;
    &--rotate {
      transform: rotate(180deg);
    }
  }

  &__input {
    height: 42px;
    width: 100%;
    padding: 0 10px;
    font-size: 14px;
    font-weight: 400;
    text-overflow: ellipsis;
    overflow: hidden;
    padding: 0 20px 0 10px;
    border-radius: 4px;
    color: #fff;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
    cursor: pointer;
    background: rgba(36, 47, 61, 0.5);
    border: 1px solid #65b9f4;
    &:focus {
      border-color: #e7ecf5;
    }
    &:disabled {
      border-color: #6c7883;
      cursor: default;
      opacity: 0.35;
    }
  }
  &__content {
    position: absolute;
    top: 41px;
    width: 100%;
    border: 1px solid #6c7883;
    box-shadow: 0px -8px 34px 0px rgba(0, 0, 0, 0.05);
    overflow: hidden;
    border-radius: 0 0 4px 4px;
    z-index: 3;
    color: #a7bed3;
    background-color: #242f3d;
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
        color: #e7ecf5;
        background-color: #6c7883;
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
    height: 24px;
    width: 109px;
  }
  &--small &__input {
    height: 24px;
    font-size: 12px;
    padding: 0 15px 0 5px;
    line-height: 24px;
  }
  &--small &__icon {
    top: 0px;
  }
  &--small &__content {
    width: auto;
    top: 23px;
    min-width: 109px;
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
}
</style>
