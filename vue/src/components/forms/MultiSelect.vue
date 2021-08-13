<template>
  <div :class="['t-multi-select', { 't-inline': inline }]" v-click-outside="outside">
    <label class="t-multi-select__label">{{ label }}</label>
    <div class="t-multi-select__input">
      <!-- <i v-show="input" class="icon icon-chevron-left" @click="next(-1)"></i> -->
      <span :class="['t-multi-select__input--text', { 't-multi-select__input--active': input }]" :title="input" @click="show = true">
        {{ input || 'Не выбрано' }}
      </span>
      <!-- <i v-show="input" class="icon icon-chevron-right" @click="next(1)"></i> -->
    </div>
    <div class="t-multi-select__content" v-show="show">
      <div v-if="filterList.length" class="t-multi__item">
        <span :class="['t-multi__item--check', { active: checkAll }]" @click="select(checkAll)" />
        <span class="t-multi__item--title">Выбрать все</span>
      </div>
      <template v-for="(item, i) in filterList">
        <div class="t-multi__item" :key="i" :title="item.label">
          <span :class="['t-multi__item--check', { active: active(item) }]" @click="select(item)"></span>
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
    id: Number,
  },
  data: () => ({
    selected: [],
    show: false,
    pagination: 0,
  }),
  created() {
    // this.$emit("selected", { name: this.value });
  },
  computed: {
    input() {
      return this.selected.map(item => item.label).join();
    },
    checkAll() {
      return this.filterList.length === this.selected.length;
    },
    filterList() {
      return this.lists.filter(item => !item.id || item.id === this.id);
    },
  },
  methods: {
    active({ value }) {
      return !!this.selected.find(item => item.value === value);
    },
    outside() {
      if (this.show) {
        this.show = false;
      }
    },
    select(list) {
      if (typeof list === 'boolean') {
        this.selected = this.filterList.map(item => (!list ? item : null)).filter(item => item);
      } else {
        if (this.selected.find(item => item.value === list.value)) {
          this.selected = this.selected.filter(item => item.value !== list.value);
        } else {
          this.selected = [...this.selected, list];
        }
      }
      this.$emit('change', this.selected)
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
    padding: 0;
    font-size: 0.875rem;
    font-weight: 400;
    width: 100%;
    display: flex;
    align-items: center;
    &--text {
      // text-align: center;
      flex-grow: 1;
      text-overflow: ellipsis;
      white-space: nowrap;
      overflow: hidden;
      color: #747b82;
      font-size: 0.75rem;
      padding: 0 8px;
    }
    &--active {
      color: #fff;
    }
    i {
      opacity: 0;
      transition: opacity 0.3s ease-in-out;
      cursor: pointer;
    }
    &:hover i {
      opacity: 1;
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
  & .t-multi-select__label {
    width: auto;
    padding: 0 20px 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    font-size: 0.75rem;
  }
  & .t-multi-select__input {
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