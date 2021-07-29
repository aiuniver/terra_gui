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
    <div v-show="select === 'GoogleDrive'" class="tabs__item">
      <Autocomplete
        :list="list"
        label="Выберите файл из Google-диска"
        @focus="focus"
        @selected="selected"
      />
    </div>
    <div v-show="select === 'URL'" class="tabs__item">
      <TInput label="Введите URL на архив исходников" @blur="blur" />
    </div>
    
  </div>
</template>

<script>
import TInput from "@/components/forms/TInput";
import Autocomplete from "@/components/forms/Autocomplete2.vue";
export default {
  name: "DatasetTab",
  components: {
    TInput,
    Autocomplete,
  },
  props: {},
  data: () => ({
    select: 'GoogleDrive',
    list: [],
    items: [
      { title: "Google drive", active: true, mode: "GoogleDrive" },
      { title: "URL-ссылка", active: false, mode: "URL" },
    ],
  }),
  methods: {
    async focus() {
      const data = await this.$store.dispatch("axios", {
        url: "/datasets/sources/?term=",
      });
      if (!data) {
        return;
      }
      console.log(data)
      this.list = data
    },
    selected({ value }) {
      this.$emit('select', { mode: 'GoogleDrive', value })
    },
    blur(value) {
      this.$emit('select', { mode: 'URL', value})
    },
    click(mode) {
      this.select = mode
      this.items = this.items.map((item) => {
        return {...item, active: (item.mode === mode) }
      })
    }
  }
};
</script>

<style lang="scss" scoped>
.tabs {
  margin-bottom: 10px;
  &__list {
    margin-bottom: 10px;
    background-color: #0e1621;
    display: block;
    padding: 1px 0 0 0;
    list-style: none;
    display: -webkit-box;
    display: -moz-box;
    display: -ms-flexbox;
    display: -webkit-flex;
    display: flex;
    -webkit-box-direction: normal;
    -moz-box-direction: normal;
    -webkit-box-orient: horizontal;
    -moz-box-orient: horizontal;
    -webkit-flex-direction: row;
    -ms-flex-direction: row;
    flex-direction: row;
    -webkit-flex-wrap: wrap;
    -ms-flex-wrap: wrap;
    flex-wrap: wrap;
    -webkit-box-pack: start;
    -moz-box-pack: start;
    -webkit-justify-content: flex-start;
    -ms-flex-pack: start;
    justify-content: flex-start;
    -webkit-align-content: flex-start;
    -ms-flex-line-pack: start;
    align-content: flex-start;
    -webkit-box-align: start;
    -moz-box-align: start;
    -webkit-align-items: flex-start;
    -ms-flex-align: start;
    align-items: flex-start;
    &--item {
      line-height: 1;
      padding: 10px 20px;
      font-weight: 600;
      font-size: 0.875rem;
      cursor: pointer;
      user-select: none;
    }
  }
  &__item{
    padding: 0 20px;
  }
}
</style>