<template>
  <main class="page-projects">
    <div class="wrapper">
      <h2>Мои проекты</h2>
      <NewModalCreateProject 
        :loading="loading" 
        @create="createProject" 
        :dialog="dialogCreate" 
        @close="dialogCreate = false" 
      />
      <NewModalEditProject
        :loading="loading" 
        @edit="editProject" 
        :dialog="dialogEdit" 
        @close="dialogEdit = false"
        :project="selectProject" 
      />
      <NewModalConfirm 
        :loading="loading" 
        @confirm="deleteProject" 
        :dialog="dialogDelete" 
        @close="dialogDelete = false" 
        :data="selectProject" 
        :actions="{ confirm: { action: 'confirm', value: 'Удалить' } }"
      >
        <template slot="headline">
          Вы собираетесь удалить проект
        </template>
        <template slot="text">
          Проект будет удален безвозвратно и вы не сможете его восстановить.
        </template>
      </NewModalConfirm>
      <div class="projects">
        <CardCreateProject @click.native="closeDialogs(), dialogCreate = true"/>
        <CardProject 
          v-bind="project" 
          v-for="(project, i) in projects" 
          :key="project.headline + i" 
          @deleteProject="closeDialogs(), dialogDelete = true" 
          @editProject="closeDialogs(), dialogEdit = true" 
          @click.native="activeProject(project)" 
        />
      </div>
    </div>
  </main>
</template>

<script>
import CardProject from '@/components/projects/CardProject'
import NewModalCreateProject from '@/components/projects/modals/NewModalCreateProject'
import NewModalConfirm from '@/components/projects/modals/NewModalConfirm'
import NewModalEditProject from '@/components/projects/modals/NewModalEditProject'
import CardCreateProject from '@/components/projects/CardCreateProject'
export default {
  name: 'Projects',
  components:{
    CardProject,
    CardCreateProject,
    NewModalCreateProject,
    NewModalConfirm,
    NewModalEditProject
  },
  data: () => ({
    dialogCreate: false,
    dialogDelete:false,
    dialogEdit: false,
    loading: false,
    selectProject: {},
    projects: [
      {
        id: 1,
        image: 'https://www.zastavki.com/pictures/1920x1080/2013/Fantasy__038385_23.jpg', 
        active: false,
        created:'17 апреля 2021', 
        edited: '3 дня назад', 
        headline: 'Проект 1. Название максимум одна ст'
      },
      {
        id: 2,
        image: 'https://www.zastavki.com/pictures/1920x1080/2013/Fantasy__038385_23.jpg', 
        active: false,
        created:'17 апреля 2021', 
        edited: '3 дня назад', 
        headline: 'Проект 1. Название максимум одна ст'
      },
      {
        id: 3,
        image: 'https://www.zastavki.com/pictures/1920x1080/2013/Fantasy__038385_23.jpg', 
        active: false,
        created:'17 апреля 2021', 
        edited: '3 дня назад', 
        headline: 'Проект 1. Название максимум одна ст'
      }
    ]
  }),
  methods:{
    closeDialogs(){
      this.dialogCreate = false
      this.dialogDelete = false
      this.dialogEdit= false
    },
    createProject(project){
      console.log('Create project', project)
    },
    editProject(project){
      this.projects = this.projects.map(el => {
        console.log(el.id === project.id)
        if(el.id === project.id) return project
        return el
      })
      console.log('Edited project', project)
    },
    deleteProject(project){
      console.log('Delete project',project)
    },
    activeProject(project){
      this.projects = this.projects.map(el => {
        return {
          ...el,
          active: el.id === project.id ? true : false
        }
      })
      this.selectProject = this.projects.find(el => el.id === project.id)
    }
  }
};
</script>

<style lang="scss" scoped>

.wrapper{
  background: #17212B;
  border-top: 0;
  height: 100%;
  padding: 20px;
  h2{
    margin-bottom: 20px;
    font-family: "Open Sans", sans-serif;
    font-size: 14px;
    font-weight: normal;
  }
}

.projects{
  display: grid;
  grid-template-columns:repeat(auto-fill, 300px);
  grid-gap: 30px;
}

</style>
