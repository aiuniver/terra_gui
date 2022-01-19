<template>
  <div
    :class="['d-input', { 'd-input--active': show }, { 'd-input--small': small, 'd-input--disabled': isDisabled }]"
    v-outside="outside"
  >
    <input
      class="d-input__input"
      v-model="input"
      :name="name"
      :disabled="disabled"
      :placeholder="placeholder || ''"
      :autocomplete="'off'"
      @click="click"
      @blur="select(false)"
      @focus="$emit('focus', $event), $event.target.select()"
    />
    <label :for="name">{{ inputLabel }}</label>
    <div class="d-input__btn">
      <div :class="['d-input__btn--down', { 'd-input__btn--disabled': isDisabled }]" @click="click">
        <i class="ci-icon ci-caret_down" />
      </div>
    </div>
    <div class="d-input__content" v-show="show">
      <div class="d-input__content--item" v-for="(item, i) in filterList" :key="i" @mousedown="select(item)">
        {{ item.label }}
      </div>
      <div class="d-input__content--empty" v-if="!filterList.length">Нет данных</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'd-auto-complete',
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
    this.input = this.value;
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
      return this.list
        ? this.list.filter(item => {
            const search = !this.all ? this.input : '';
            return search ? item.label.toLowerCase().includes(search.toLowerCase()) : true;
          })
        : [];
    },
    search: {
      set(value) {
        this.input = value;
      },
      get() {
        const list = this.list ?? [];
        const label = list.find(item => item.value === this.selected?.value || item.value === this.value)?.label || '';
        return label || '';
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
        this.search = this.selected.label || this.value || '';
      }
      this.show = false;
    },
    click(e) {
      if (this.isDisabled) return
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
    }
  },
};
</script>

<style lang="scss" scoped>
@import '@/assets/scss/components/fields.scss';
.d-input {
  position: relative;
  height: 42px;
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
  label {
    position: absolute;
    margin-left: 10px;
    margin-top: 6px;
    font-size: 12px;
    line-height: 12px;
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
    cursor: pointer;
  }
  &__content {
    position: absolute;
    top: 41px;
    width: 100%;
    overflow: auto;
    border-radius: 0 0 4px 4px;
    z-index: 3;
    background: #242F3D;
    box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.25);
    border-radius: 4px;
    &--item {
      color: inherit;
      font-size: 14px;
      line-height: 24px;
      text-align: left;
      cursor: pointer;
      text-overflow: ellipsis;
      white-space: nowrap;
      overflow: hidden;
      padding: 8px 10px;
      &:first-child {
        padding: 12px 10px 8px;
      }
      &:last-child {
        padding: 8px 10px 12px;
      }
      &:hover {
        background: #1E2734;
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
