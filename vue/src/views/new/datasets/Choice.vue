<template>
  <div class="page-choice">
    <div class="page-choice__menu">
      <ChoiceMenu @select="selectedType = $event" 
      :selectedType="selectedType" 
      :selectedTag="selectedTag"
      @tagClick="handleTag"/>
    </div>
    <div class="page-choice__main">
      <Datasets :datasets="filteredList" :selectedType="selectedType" @choice="getVersions" />
    </div>
    <at-modal
      v-model="showModal"
      :showConfirmButton="false"
      :showCancelButton="false"
      :width="450"
      title="Выбор версии"
      @on-cancel="onCancel"
    >
      <div v-for="(item, idx) in versions"
      class="page-choice__versions"
      :key="idx" 
      @click="setVersion(item)"
      :class="{ 'active': item.alias === selectedVersion.alias }"
      >
        <span class="name">{{ item.name }}</span>
        <span class="info">{{ getSize(item.size) }}</span>
        <span class="info">{{ getDate(item.date) }}</span>
        <i v-if="selectedSet.group === 'custom'" @click.stop="deleteVersion(item)" class="t-icon icon-modeling-remove"></i>
      </div>
      <template v-slot:footer>
        <d-button
          @click="setChoice" 
          :disabled="!selectedVersion.alias" 
          style="flex-basis: 50%;">Выбрать</d-button>
        <d-button 
          @click="createVersion" 
          v-if="selectedSet.group === 'custom'" 
          color="secondary" 
          direction="left" 
          style="flex-basis: 50%;">Создать версию</d-button>
      </template>
    </at-modal>
  </div>
</template>

<script>
import Datasets from '@/components/datasets/choice/'
import ChoiceMenu from '@/components/datasets/choice/ChoiceMenu'
import { mapGetters } from 'vuex'

export default {
  name: 'Choice',
  components: {
    Datasets,
    ChoiceMenu
  },
  data: () => ({
    selectedType: 0,
    showModal: false,
    versions: [],
    tID: null,
    selectedSet: {},
    selectedVersion: {},
    selectedTag: {},
    versionTID: null
  }),
  computed: {
    ...mapGetters({
      datasets: 'datasets/getDatasets'
    }),
    filteredList() {
      let datasets = this.datasets
      if (this.selectedType !== 0) {
        if (this.selectedType === 1) {
          datasets = datasets.filter(item => this.$store.getters['datasets/getRecent'].includes(item.id))
        }
        if (this.selectedType === 2) {
          const projectSet = this.$store.getters['projects/getProject'].dataset || {}
          datasets = datasets.filter(item => item.id === `${projectSet.group}_${projectSet.alias}`)
        }
        if (this.selectedType === 3) {
          datasets = datasets.filter(item => item.group === 'terra')
        }
      }
      if (!this.selectedTag.type) return datasets
      return datasets.filter(item => {
        if (this.selectedTag.type === 'group') return this.selectedTag.alias === item.architecture
        return item.tags.includes(this.selectedTag.alias) && this.selectedTag.group === item.architecture
      })
    }
  },
  methods: {
    setVersion(item) {
      if (this.selectedVersion.alias === item.alias) return this.selectedVersion = {}
      this.selectedVersion = item
    },
    async createVersion() {
      let res
      res = await this.$store.dispatch('axios', { url: '/datasets/create/version/', data: {
        alias: this.selectedSet.alias,
        group: this.selectedSet.group,
        ...(this.selectedVersion.alias && { version: this.selectedVersion.alias })
      } })
      if (res.success) {
        this.$store.dispatch('settings/setOverlay', true)
        this.versionProgress()
      }
    },
    versionProgress() {
      this.versionTID = setInterval(async () => {
        const res = await this.$store.dispatch('datasets/versionProgress')
        if (res.data?.finished) {
          clearInterval(this.versionTID)
          await this.$store.dispatch('projects/get')
          this.$router.push('/create')
          this.$store.dispatch('settings/setOverlay', false)
        }
      }, 1000)
    },
    getDate(val) {
      if (!val) return ''
      return new Date(val).toLocaleString('ru-RU', { dateStyle: 'short', timeStyle: 'short' })
    },
    getSize(size) {
      if (size?.value) return `${size.short.toFixed(2)} ${size.unit}`
      return ''
    },
    onCancel() {
      this.versions = []
      this.selectedVersion = {}
      this.selectedSet = {}
    },
    async getVersions({ alias, group }) {
      this.showModal = true
      const { data } = await this.$store.dispatch('axios', { url: '/datasets/versions/', data: { group, alias } })
      this.selectedSet = { group, alias }
      this.versions = data
    },
    async setChoice() {
      this.$store.dispatch('settings/setOverlay', true);
      const { group, alias } = this.selectedSet
      const { success } = await this.$store.dispatch('datasets/choice', { group, alias, version: this.selectedVersion.alias })

      if (success) {
        this.$store.dispatch('messages/setMessage', { message: `Загружаю датасет` });
        this.createInterval();
      }
    },
    createInterval() {
      this.tID = setTimeout(async () => {
        const res = await this.$store.dispatch('datasets/choiceProgress', {});
        if (res) {
          const { data } = res;
          if (data) {
            const { finished, message, percent, error } = data;
            this.$store.dispatch('messages/setProgressMessage', message);
            this.$store.dispatch('messages/setProgress', percent);
            if (finished) {
              this.$store.dispatch('messages/setProgress', 0);
              this.$store.dispatch('messages/setProgressMessage', '');
              await this.$store.dispatch('projects/get');
              this.$store.dispatch('messages/setMessage', {
                message: `Датасет «${data?.data?.dataset?.name || ''}» выбран`,
              });
              this.$store.dispatch('settings/setOverlay', false);
              this.showModal = false
              this.onCancel()
            } else {
              if (error) {
                // this.$store.dispatch('messages/setMessage', { error });
                this.$store.dispatch('messages/setProgressMessage', '');
                this.$store.dispatch('messages/setProgress', 0);
                this.$store.dispatch('settings/setOverlay', false);
                return;
              }
              this.createInterval();
            }
          } else {
            this.$store.dispatch('settings/setOverlay', false);
          }
        } else {
          this.$store.dispatch('settings/setOverlay', false);
        }
      }, 1000);
    },
    handleTag(e) {
      if (e.type === this.selectedTag.type && e.alias === this.selectedTag.alias) return this.selectedTag = {};
      this.selectedTag = e;
    },
    deleteVersion(item) {
      this.$Modal.confirm({
        title: 'Внимание!',
        content: 'Уверены, что хотите удалить эту версию?',
        width: 300,
        callback: async action => {
          if (action === 'confirm') {
            await this.$store.dispatch('axios', { url: '/datasets/delete/version/', data: { 
              group: this.selectedSet.group,
              alias: this.selectedSet.alias,
              version: item.alias
             } })
             this.getVersions(this.selectedSet)
             await this.$store.dispatch('projects/get')
          }
        }
      })
    },
    async getDatasets() {
      const { data } = await this.$store.dispatch('axios', { url: '/datasets/info/' })
      data.datasets.forEach(item => {
        item.datasets.forEach(dataset => {
          this.datasets.push({
            ...dataset,
            group: item.alias
          })
        })
      })
      this.$store.commit('datasets/SET_GROUPS', data.groups)
      this.tags = data.tags
    }
  },
  created() {
    this.$store.dispatch('datasets/get')
  }
};
</script>

<style lang="scss">
@import "@/assets/scss/variables/default.scss";
.page-choice {
  position: relative;
  display: flex;
  height: 100%;
  &__menu {
    width: 250px;
    border-right: 1px solid $color-black;
    padding-top: 18px;
    flex-shrink: 0;
    hr {
      border: none;
      border-bottom: 1px solid $color-dark-gray;
      margin: 20px;
    }
  }
  &__main {
    width: 100%;
  }
  &__versions {
    font-size: 14px;
    max-height: 400px;
    overflow-y: auto;
    cursor: pointer;
    padding: 5px;
    border-bottom: 2px solid #1E2734;
    display: flex;
    gap: 10px;
    align-items: center;
    .name {
      flex-grow: 1;
      word-wrap: anywhere;
      overflow-wrap: anywhere;
    }
    .info {
      font-size: 12px;
      color: #A7BED3;
      white-space: nowrap;
    }
    &:hover, &.active {
      background: #1E2734;
      color: #65B9F4;
    }
    .t-icon {
      width: 16px;
      height: 16px;
    }
  }
}
</style>