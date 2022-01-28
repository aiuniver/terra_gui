<template>
  <div class="tabs-download">
    <div class="tabs-download-list flex align-center">
      <div
        v-for="{ text, tab } in items"
        :class="['tabs-download-list__item', { 'tabs-download-list__item--active': isActive(tab) }]"
        :key="`tab_${tab}`"
        @click="active = tab"
      >
        {{ text }}
      </div>
    </div>
    <div class="tabs-download-content mt-10">
      <t-field icon="google" label="Выберите файл на Google диске" v-if="active === 0">
        <d-auto-complete-two icon="google-drive" v-model="select" placeholder="Введите имя файла" :list="list" @click="onFocus" />
      </t-field>
      <t-field icon="link" label="Загрузите по ссылке" v-if="active === 1">
        <d-input-text v-model="url" placeholder="URL" />
      </t-field>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DatasetDownloadTabs',
  data: () => ({
    url: '',
    select: '',
    list: [],
    items: [
      { text: 'Google диск', tab: 0 },
      { text: 'URL', tab: 1 },
    ],
    active: 0,
  }),
  computed: {},
  methods: {
    async onFocus() {
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
    isActive(tab) {
      return this.active === tab;
    },
  },
};
</script>

<style lang="scss" scoped>
.tabs-download {
  &-list {
    height: 50px;
    &__item {
      width: 50%;
      text-align: center;
      position: relative;
      cursor: pointer;
      color: $color-light-gray;
      transition: color 0.3s ease;
      &:hover {
        color: $color-light-blue;
      }
      &--active {
        &::before {
          content: '';
          width: 235px;
          height: 1px;
          position: absolute;
          bottom: -5px;
          left: 50%;
          background: $color-light-blue;
          transform: translateX(-50%);
        }
      }
    }
  }
}
</style>
