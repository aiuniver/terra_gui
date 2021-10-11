<template>
  <div>
    <div :class="['t-multi-select', { 't-inline': inline }]" v-click-outside="outside">
      <label class="t-multi-select__label">
        <slot>{{ label }}</slot>
      </label>
      <div
        :class="['t-multi-select__input', { 't-multi-select__error': error }, { 't-multi-select__input--show': show }]"
      >
        <!-- <i v-show="input" class="icon icon-chevron-left" @click="next(-1)"></i> -->
        <span
          :class="['t-multi-select__input--text', { 't-multi-select__input--active': input }]"
          :title="input"
          @click="click"
        >
          {{ input || placeholder }}
        </span>
        <!-- <i v-show="input" class="icon icon-chevron-right" @click="next(1)"></i> -->
      </div>
      <div class="t-multi-select__content" v-show="show">
        <div v-if="filterList.length" class="t-multi__item" @click="select(checkAll)">
          <span :class="['t-multi__item--check', { 't-multi__item--active': checkAll }]" />
          <span class="t-multi__item--title">Выбрать все</span>
        </div>
        <template v-for="(item, i) in filterList">
          <div class="t-multi__item" :key="i" :title="item.label" @click="select(item)">
            <span :class="['t-multi__item--check', { 't-multi__item--active': active(item) }]"></span>
            <span class="t-multi__item--title">{{ item.label }}</span>
          </div>
        </template>
        <div v-if="!filterList.length" class="t-multi__item t-multi__item--empty">
          <span class="t-multi__item--title">Нет данных</span>
        </div>
      </div>
    </div>
    <MultiSelectTable v-if="selectedTable.length" :id="id" :table="selectedTable" label="Таблица" inline @change="$emit('change', $event)"/>
  </div>
</template>

<script>
import MultiSelectTable from './MultiSelectTable';
import blockMain from '@/mixins/datasets/blockMain';
export default {
  name: 't-multi-select',
  components: {
    MultiSelectTable
  },
  mixins: [blockMain],
  props: {
    name: String,
    id: Number,
    label: {
      type: String,
      default: 'Label',
    },
    // lists: {
    //   type: Array,
    //   required: true,
    //   default: () => [],
    // },
    placeholder: {
      type: String,
      default: 'Не выбрано',
    },
    disabled: Boolean,
    inline: Boolean,
    value: Array,
  },
  data: () => ({
    selected: [],
    show: false,
    pagination: 0,
  }),
  computed: {
    selectedTable() {
      return this.selected.filter(item => item.type === 'table')
    },
    lists() {
      return this.mixinFiles;
    },
    errors() {
      return this.$store.getters['datasets/getErrors'](this.id);
    },
    error() {
      const key = this.name;
      return this.errors?.[key]?.[0] || this.errors?.parameters?.[key]?.[0] || '';
    },
    input() {
      return this.selected.map(item => item.label).join();
    },
    checkAll() {
      return this.filterList.length === this.selected.length;
    },
    filterList() {
      const { type } = this.$store.getters['datasets/getInputDataByID'](this.id);
      const filter = this.mixinFilter?.[type || ''] || [];
      // console.log(type, filter);
      return (
        this.lists
          // .filter(item => (!item.id || item.id === this.id) || item.type === 'table' )
          // .filter(item => item.type !== 'table')
          .filter(item => filter.includes(item.type))
      );
    },
  },
  methods: {
    click() {
      this.show = true;
      if (this.error) {
        // console.log(this.id, this.name);
        this.$store.dispatch('datasets/cleanError', { id: this.id, name: this.name });
      }
    },
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
      // console.log(this.id)
      this.$emit('multiselect', { value: this.selected, id: this.id });
      this.mixinCheck(this.selected, this.id);
    },
  },
  created() {
    // console.log(this.value);
    // console.log(this.filterList.filter(item => item));
    const value = this.value;
    if (Array.isArray(value)) {
      this.selected = this.filterList.filter(item => value.includes(item.value));
    }
  },
  watch: {
    filterList() {
      this.selected = this.selected.filter(element => this.lists.find(item => item.value === element.value));
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
    line-height: 1;
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
    &--show {
      border-bottom-right-radius: 0px;
      border-bottom-left-radius: 0px;
    }
    &--text {
      flex-grow: 1;
      text-overflow: ellipsis;
      white-space: nowrap;
      overflow: hidden;
      color: #747b82;
      font-size: 0.75rem;
      padding: 0 5px;
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
  &__error {
    border-color: #b53b3b;
  }
  &__content {
    padding: 2px 0;
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
  padding: 1px 5px;
  align-items: center;
  cursor: pointer;
  &--empty {
    cursor: default;
  }
  &:hover {
    color: #e7ecf5;
    background-color: #6c7883;
  }
  &--empty:hover {
    background-color: #242f3d;
  }
  &--check {
    height: 10px;
    background-color: #eee;
    margin-right: 5px;
    border-radius: 2px;
    flex: 0 0 10px;
  }
  &--active {
    background-color: #5191f2;
    width: 10px;
    height: 10px;
    border: 1px solid white;
  }
  &--title {
    color: #a7bed3;
    font-size: 0.7em;
    line-height: 1.5;
    text-align: left;
    cursor: pointer;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
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
    padding: 0 10px;
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
    width: auto;
    top: 23px;
    min-width: 100px;
  }
}
</style>