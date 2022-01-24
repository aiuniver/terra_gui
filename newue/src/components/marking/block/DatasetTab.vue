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
    <div v-show="value === 'GoogleDrive'" class="tabs__title"><i></i> Выберите файл на Google диске</div>
    <div v-show="value === 'URL'" class="tabs__title">Введите URL</div>
    <div v-show="value === 'GoogleDrive'" class="tabs__item">
      <Autocomplete2
        :list="list"
        :name="'gdrive'"
        @focus="focus"
        @change="selected"
      />
    </div>
    <div v-show="value === 'URL'" class="tabs__item">
      <t-input label="" @change="change" />
    </div>
  </div>
</template>

<script>
import Autocomplete2 from "@/components/forms/Autocomplete2.vue";
export default {
  name: "MarkingTab",
  components: {
    Autocomplete2,
  },
  props: {
    value: {
      type: String,
      default: "GoogleDrive"
    },
  },
  data: () => ({
    list: [],
    items: [
      { title: "Google drive", active: true, mode: "GoogleDrive" },
      { title: "URL", active: false, mode: "URL" },
    ],
  }),
  methods: {
    async focus() {
      const { data } = await this.$store.dispatch("axios", {
        url: "/datasets/sources/",
      });
      if (!data) {
        return;
      }
      // console.log(data);
      this.list = data;
    },
    selected({ value, label }) {
      this.$emit("select", { mode: "GoogleDrive", value, label });
    },
    change(value) {
      this.$emit("select", { mode: "URL", value });
    },
    click(mode) {
      this.select = mode;
      this.$emit('input', mode)
      // this.$emit("select", {});
      this.items = this.items.map((item) => {
        return { ...item, active: item.mode === mode };
      });
    },
  },
};
</script>

<style lang="scss" scoped>
.tabs {
  &__title {
    padding: 20px 20px 0;
    font-size: 14px;
    line-height: 24px;
    display: flex;
    align-items: center;
    i {
      width: 24px;
      height: 24px;
      display: inline-block;
      margin-right: 10px;
      background-image: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTcuNzA5OSAzLjUyTDEuMTQ5OSAxNUw0LjU2OTkgMjAuOTlMMTEuMTI5OSA5LjUyTDcuNzA5OSAzLjUyWk0xMy4zNDk5IDE1SDkuNzI5OUw2LjI5OTkgMjFIMTQuNTM5OUMxMy41Nzk5IDE5Ljk0IDEyLjk5OTkgMTguNTQgMTIuOTk5OSAxN0MxMi45OTk5IDE2LjMgMTMuMTI5OSAxNS42MyAxMy4zNDk5IDE1Wk0xOS45OTk5IDE2VjEzSDE3Ljk5OTlWMTZIMTQuOTk5OVYxOEgxNy45OTk5VjIxSDE5Ljk5OTlWMThIMjIuOTk5OVYxNkgxOS45OTk5Wk0yMC43MDk5IDExLjI1TDE1LjQxOTkgMkg4LjU3OTlWMi4wMUwxNC43Mjk5IDEyLjc4QzE1LjgxOTkgMTEuNjggMTcuMzI5OSAxMSAxOC45OTk5IDExQzE5LjU4OTkgMTEgMjAuMTY5OSAxMS4wOSAyMC43MDk5IDExLjI1WiIgZmlsbD0iI0E3QkVEMyIvPgo8L3N2Zz4K');
    }
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