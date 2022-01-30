<template>
  <main class="page-projects">
    <div class="wrapper">
      <h2>Мои проекты</h2>
      <div class="projects">
        <CardCreateProject @click.native="closeDialogs(), (dialogCreate = true)" />
        <CardProject
          v-for="(project, i) in projects"
          v-bind="project"
          :key="project.label + i"
          @deleteProject="closeDialogs(), (dialogDelete = true)"
          @editProject="closeDialogs(), (dialogEdit = true)"
          @load="onLoad(project)"
        />
      </div>
    </div>
    <d-modal v-model="dialogCreate" title="Мой профиль">
      <t-field label="Название проекта *">
        <d-input-text v-model="name" placeholder="Введите название проекта" />
      </t-field>
      <t-field label="Перезаписать">
        <d-checkbox v-model="overwrite" />
      </t-field>
      <d-upload />
      <template slot="footer">
        <d-button color="primary" @click="onSave({ name, overwrite })" :disabled="isSave" >Сохранить</d-button>
        <d-button color="secondary" direction="left" @click="dialogCreate = false">Отменить</d-button>
      </template>
    </d-modal>
    <d-modal v-model="dialogLoad" title="Загрузить проект">
      <template slot="footer">
        <d-button color="primary" @click="loadProject">Загрузить</d-button>
        <d-button color="secondary" direction="left" @click="dialogLoad = false">Отменить</d-button>
      </template>
    </d-modal>
  </main>
</template>

<script>
import CardProject from '@/components/projects/CardProject';
import CardCreateProject from '@/components/projects/CardCreateProject';
import { debounce } from '@/utils/core/utils';
import { mapActions, mapGetters } from 'vuex';
export default {
  name: 'Projects',
  components: {
    CardProject,
    CardCreateProject,
  },
  data: () => ({
    name: '',
    overwrite: false,
    selected: {},
    show: true,
    list: [],
    debounce: null,
    dialogCreate: false,
    dialogDelete: false,
    dialogEdit: false,
    dialogLoad: false,
    loading: false,
    selectProject: {},
    tempProject: {},
  }),
  computed: {
    ...mapGetters({
      projects: 'projects/getProjectsList',
    }),
    isSave() {
      return Boolean(!this.name)
    }
  },
  methods: {
    ...mapActions({
      infoProject: 'projects/infoProject',
    }),
    async progress() {
      const res = await this.$store.dispatch('projects/progress', {});
      // console.log(res?.data?.progress)
      if (res && res?.data) {
        const { finished, message, percent } = res.data;
        this.$store.dispatch('messages/setProgressMessage', message);
        this.$store.dispatch('messages/setProgress', percent);
        if (!finished) {
          this.debounce(true);
        } else {
          this.$store.dispatch('projects/get');
          this.$store.dispatch('settings/setOverlay', false);
          this.$emit('message', { message: `Проект загружен` });
          this.dialog = false;
        }
      }
      if (res?.error) this.$store.dispatch('settings/setOverlay', false);
    },
    remove(list) {
      // this.show = false;
      this.$Modal
        .confirm({
          title: 'Удаление проекта',
          content: `Вы действительно хотите удалить проект «${list.label}»?`,
          width: 300,
          maskClosable: false,
          showClose: false,
        })
        .then(() => {
          this.removeProject(list);
          // this.show = true;
        })
        .catch(() => {
          // this.show = true;
        });
    },
    async onSave(data) {
      try {
        this.loading = true;
        const res = await this.$store.dispatch('projects/saveProject', data);
        if (res && !res.error) {
          this.dialog = false;
          this.overwrite = false;
        }
        this.dialogCreate = false;
      } catch (error) {
        console.log(error);
        this.loading = false;
      }
    },
    onLoad(project) {
      this.tempProject = project;
      this.dialogLoad = true;
    },
    async loadProject() {
      this.dialogLoad = false;
      try {
        const res = await this.$store.dispatch('projects/load', { value: this.tempProject.value });
        console.log(res);
        if (res?.success) {
          this.$store.dispatch('settings/setOverlay', true);
          this.debounce(true);
        }
      } catch (error) {
        console.log(error);
      }
    },
    async removeProject(list) {
      console.log(list);
      try {
        const res = await this.$store.dispatch('projects/remove', { path: list.value });
        if (res && !res.error) {
          this.$emit('message', { message: `Проект «${list.label}» удален` });
          await this.infoProject();
        }
      } catch (error) {
        console.log(error);
      }
    },
    closeDialogs() {
      this.dialogCreate = false;
      this.dialogDelete = false;
      this.dialogEdit = false;
    },
    createProject(project) {
      console.log('Create project', project);
    },
    editProject(project) {
      this.projects = this.projects.map(el => {
        console.log(el.id === project.id);
        if (el.id === project.id) return project;
        return el;
      });
      console.log('Edited project', project);
    },
    deleteProject(project) {
      console.log('Delete project', project);
    },
  },
  created() {
    this.debounce = debounce(status => {
      if (status) {
        this.progress();
      }
    }, 1000);
    this.debounce(this.isLearning);
  },
  beforeDestroy() {
    this.debounce(false);
  },
  mounted() {
    this.infoProject();
  },
};
</script>

<style lang="scss" scoped>
.wrapper {
  // background: #17212b;
  border-top: 0;
  height: 100%;
  padding: 20px;
  h2 {
    margin-bottom: 20px;
    font-family: 'Open Sans', sans-serif;
    font-size: 14px;
    font-weight: normal;
  }
}

.projects {
  display: grid;
  grid-template-columns: repeat(auto-fill, 300px);
  grid-gap: 30px;
}
</style>
