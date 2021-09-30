<template>
  <div class="tabs">
    <ul class="tabs__list">
      <li
        v-for="({ title, active, mode }, i) of items"
        :key="i"
        :class="['tabs__list--item', { active }]"
        @click.prevent="click(mode)"
      >
        {{ title }}
      </li>
    </ul>
    <div class="tabs__title">Создание датасета</div>
    <div v-show="value === 'GoogleDrive'" class="tabs__item">
      <Autocomplete2
        :list="list"
        :name="'gdrive'"
        label="Выберите файл из Google-диске"
        @focus="focus"
        @change="selected"
      />
    </div>
    <div v-show="value === 'URL'" class="tabs__item">
      <t-input label="Введите URL на архив исходников" @input="change" />
    </div>
  </div>
</template>

<script>
import Autocomplete2 from '@/components/forms/Autocomplete2.vue';
export default {
  name: 'DatasetTab',
  components: {
    Autocomplete2,
  },
  props: {
    value: {
      type: String,
      default: 'GoogleDrive',
    },
  },
  data: () => ({
    list: [],
    items: [
      { title: 'Google drive', active: true, mode: 'GoogleDrive' },
      { title: 'URL', active: false, mode: 'URL' },
    ],
  }),
  methods: {
    async focus() {
      const { data } = await this.$store.dispatch('axios', {
        url: '/datasets/sources/',
      });
      if (!data) {
        return;
      }
      // console.log(data);
      this.list = data;
    },
    selected({ value, label }) {
      this.$emit('select', { mode: 'GoogleDrive', value, label });
    },
    change(value) {
      this.$emit('select', { mode: 'URL', value: value ? value.trim() : '' });
    },
    click(mode) {
      this.select = mode;
      this.$emit('input', mode);
      // this.$emit("select", {});
      this.items = this.items.map(item => {
        return { ...item, active: item.mode === mode };
      });
    },
  },
};
</script>

<style lang="scss" scoped>
.tabs {
  &__title {
    padding: 20px;
    font-size: 14px;
    line-height: 24px;
    display: flex;
    align-items: center;
  }
  &__list {
    background-color: #0e1621;
    padding: 1px 0 0 0;
    list-style: none;
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: flex-start;
    align-content: flex-start;
    align-items: flex-start;
    &--item {
      flex: 1;
      padding: 10px 20px;
      cursor: pointer;
      -webkit-user-select: none;
      -moz-user-select: none;
      -ms-user-select: none;
      user-select: none;
      border-radius: 5px 5px 0px 0px;
      font-size: 14px;
      line-height: 19px;
      text-align: center;
    }
  }
  &__item {
    padding: 0 20px;
  }
}
</style>
