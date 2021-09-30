<template>
  <div :class="['t-select', { 't-select--active': show }]">
    <label class="t-select__label">{{ label }}</label>
    <i :class="['t-icon icon-file-arrow', { spin: show }]" @click="click"></i>
    <input
      class="t-select__input"
      v-model="search"
      readonly
      :name="name"
      :disabled="disabled"
      :placeholder="placeholder || ''"
      :autocomplete="'off'"
      @click="click"
      @blur="select(false)"
    />
    <div class="t-select__content" v-show="show">
      <div class="t-select__content--item" v-for="(item, i) in filterList" :key="i" @mousedown="select(item)">
        {{ item.label }}
      </div>
      <div class="t-select__content--empty" v-if="!filterList.length">Нет данных</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-select-new',
  props: {
    label: {
      type: String,
      default: 'Label',
    },
    type: String,
    placeholder: String,
    value: [String, Number],
    name: String,
    parse: String,
    list: [Array, Object],
    disabled: Boolean,
    error: String,
  },
  data() {
    return {
      selected: {},
      show: false,
      search: '',
    };
  },
  created() {
    // this.$emit("selected", { name: this.value });
    // this.$emit('parse', { parse: this.parse, value: this.value });
    this.search = this.value;
  },
  computed: {
    filterList() {
      return this.list || [];
      // ? this.list.filter(item => {
      //     const search = this.search;
      //     return search ? item.label.toLowerCase().includes(search.toLowerCase()) : true;
      //   })
      // : [];
    },
  },
  methods: {
    select(item) {
      // console.log(item);
      if (item) {
        this.selected = item;
        this.show = false;
        this.search = item.label;
        this.$emit('input', this.selected.value);
        this.$emit('change', { name: this.name, value: item.value });
        this.$emit('parse', { name: this.name, parse: this.parse, value: item.value });
      } else {
        this.search = this.selected.label || this.value;
        this.show = false;
      }
    },
    click() {
      // target.select();
      this.show = true;
      this.$emit('focus', true);
    },
  },
  watch: {
    value: {
      handler(value) {
        // console.log(value)
        this.show = false;
        this.search = value;
      },
    },
  },
};
</script>

<style lang="scss" scoped>
.t-select {
  position: relative;
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  margin-bottom: 10px;
  align-items: center;
  & .t-icon{
    position: absolute;
    left: 94px;
    width: 8px;
    cursor: pointer;
    transition-duration: 100ms;
  }
  & .spin{
    transform: rotate(180deg);
  }
  &--active {
    .t-select__input {
      border-radius: 4px 4px 0 0;
    }
    // .t-select__content {
      // border-top: 0 !important;
    // }
  }
  &__label {
    // width: 150px;
    // max-width: 130px;
    padding: 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    // white-space: nowrap;
    overflow: hidden;
  }
  &__input {
    flex: 0 0 109px;
    height: 24px;
    font-size: 0.75rem;
    max-width: 109px;
    text-overflow: ellipsis;
    overflow: hidden;
    padding: 0 20px 0 5px;
    font-weight: 400;
    border: 1px solid;
    border-radius: 4px;
    border-color: #6c7883;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
    color: #fff;
    background: #242f3d;
    cursor: pointer;
    &:focus {
      border: 1px solid #e7ecf5;
    }
    &:disabled {
      border: 1px solid #6c7883;
      cursor: auto;
      opacity: 0.35;
    }
  }
  &__content {
    position: absolute;
    top: 23px;
    background-color: #242f3d;
    width: auto;
    min-width: 109px;
    border: 1px solid #6c7883;
    box-shadow: 0px -8px 34px 0px rgba(0, 0, 0, 0.05);
    overflow: hidden;
    border-radius: 0 0 4px 4px;
    z-index: 3;
    &--item {
      color: #a7bed3;
      font-size: 0.75em;
      line-height: 1.5;
      text-align: left;
      cursor: pointer;
      text-overflow: ellipsis;
      white-space: nowrap;
      overflow: hidden;
      padding: 2px 5px;
      &:hover {
        color: #e7ecf5;
        background-color: #6c7883;
      }
    }
    &--empty {
      color: #a7bed3;
      font-size: 0.7em;
      line-height: 1.5;
      padding: 2px 5px;
      cursor: default;
    }
  }
}
</style>
