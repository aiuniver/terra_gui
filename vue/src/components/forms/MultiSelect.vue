<template>
  <div :class="['t-multi-select', { 't-inline': inline }]" v-click-outside="outside">
    <label class="t-multi-select__label">{{ label }}</label>
    <input
      class="t-multi-select__input"
      v-model="search"
      :name="name"
      :disabled="disabled"
      :placeholder="placeholder"
      @focus="focus"
      @blur="select(false)"
    />
    <div class="t-multi-select__content" v-show="show">
      <div v-if="filterList.length" class="t-multi__item">
        <span
          :class="['t-multi__item--check', { active: checkAll }]"
          @click="(checkAll = !checkAll), $emit('checkAll', checkAll)"
        ></span>
        <span class="t-multi__item--title">Выбрать все</span>
      </div>
      <template v-for="(item, i) in filterList">
        <div class="t-multi__item" :key="i" :title="item.label">
          <span :class="['t-multi__item--check', { active: item.active }]" @click="$emit('check', item)"></span>
          <span class="t-multi__item--title">{{ item.label }}</span>
        </div>
      </template>
      <div v-if="!filterList.length" class="t-multi__item">
        <span class="t-multi__item--title">Нет данных</span>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'TMultiSelect',
  props: {
    name: String,
    lists: {
      type: Array,
      required: true,
      default: () => [],
    },
    placeholder: String,
    disabled: Boolean,
    label: {
      type: String,
      default: 'Label',
    },
    inline: Boolean,
    value: String,
    sloy: Number,
  },
  data() {
    return {
      selected: {},
      show: false,
      search: '',
      checkAll: false,
    };
  },
  created() {
    // this.$emit("selected", { name: this.value });
    this.search = this.value;
  },
  computed: {
    filterList() {
      return this.lists
        ? this.lists.filter(item => {
            return item.active !== true || item.sloy === this.sloy;
          })
        : [];
      // return this.lists
      //   ? this.lists.filter(item => {
      //       const search = this.search;
      //       return search ? item.label.toLowerCase().includes(search.toLowerCase()) : true;
      //     })
      //   : [];
    },
  },
  methods: {
    outside() {
      if (this.show) {
        this.show = false;
      }
    },
    check(i) {
      console.log(i);
    },
    select(item) {
      // console.log(item);
      if (item) {
        this.selected = item;
        // this.show = false;
        this.search = item.label;
        this.$emit('input', this.selected.value);
        this.$emit('change', item);
      } else {
        this.search = this.selected.label;
        // this.show = false;
      }
    },
    focus() {
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
.t-multi-select {
  position: relative;
  margin-bottom: 10px;
  &__label {
    width: 150px;
    max-width: 130px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0 0 10px 0;
    line-height: 1.5;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
  }
  &__input {
    color: #fff;
    border: 1px solid #6c7883;
    border-radius: 4px;
    background: #242f3d;
    height: 42px;
    padding: 0 10px;
    font-size: 0.875rem;
    font-weight: 400;
    width: 100%;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
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
    background-color: #242f3d;
    width: 100%;
    max-height: 190px;
    border: 1px solid #6c7883;
    box-shadow: 0px -8px 34px 0px rgb(0 0 0 / 5%);
    overflow: auto;
    border-radius: 0 0 4px 4px;
    z-index: 102;
  }
}
.t-multi__item {
  display: flex;
  padding: 2px 6px;
  align-items: center;
  &:hover {
    color: #e7ecf5;
    background-color: #6c7883;
  }
  &--check {
    cursor: pointer;
    height: 10px;
    width: 10px;
    background-color: #eee;
    margin-right: 5px;
    border-radius: 2px;
    &.active {
      background-color: #5191f2;
      width: 10px;
      height: 10px;
      border: 1px solid white;
    }
  }
  &--title {
    color: #a7bed3;
    font-size: 0.7em;
    line-height: 1.5;
    text-align: left;
  }
}
.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  align-items: center;
  & label {
    width: auto;
    padding: 0 20px 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    font-size: 0.75rem;
  }
  & input {
    height: 24px;
    font-size: 12px;
    width: 100px;
  }
  & .t-multi-select__content {
    width: 100px;
    top: 24px;
  }
}
</style>